import cv2
from djitellopy import Tello
from collections import deque
import numpy as np
import time
import pickle
from ultralytics import YOLO
import argparse

def findPerson(img, model):
    '''
    output:
        img - image overlay of tracking target
        info[0] - (x, y) coordinate of center of target in frame
        info[1] - area of target in frame
    '''
    # Run YOLOv8 inference
    results = model(img, classes=[0])  # 0 is the class ID for 'person'
    
    # Get the first result
    result = results[0]
    boxes = result.boxes
    
    myPersonListC = []
    myPersonListArea = []
    
    # Process each detected person
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Calculate center and area
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        area = (x2 - x1) * (y2 - y1)
        
        # Display confidence score
        conf = box.conf[0].item()
        cv2.putText(img, f'Conf: {conf:.2f}', (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myPersonListC.append([cx, cy])
        myPersonListArea.append(area)
    
    if len(myPersonListArea) != 0:
        # Return the person with the largest area
        i = myPersonListArea.index(max(myPersonListArea))
        return img, [myPersonListC[i], myPersonListArea[i]]
    else:
        return img, [[0, 0], 0]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Person detection with YOLOv8 on Tello drone')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                       help='YOLO model to use (default: yolov8n.pt)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Target frames per second for processing (default: 10)')
    args = parser.parse_args()

    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO(args.model)
    print(f"Model loaded successfully! Using model: {args.model}")

    try:
        tello = Tello()
        tello.connect()
        print(f"Tello battery: {tello.get_battery()}%")
        
        # Start video stream (Tello's native resolution is 720x480)
        print("Starting video stream...")
        tello.streamon()
        time.sleep(2)  # Give time for stream to initialize

        # Frame rate control
        frame_time = 1.0 / args.fps
        last_frame_time = time.time()
        frame_count = 0
        fps = 0
        last_fps_time = time.time()
        
        # Initialize frame reader
        frame_reader = tello.get_frame_read()
        last_frame = None

        while True:
            current_time = time.time()
            
            # Skip frame if not enough time has passed
            if current_time - last_frame_time < frame_time:
                continue
                
            # Update frame timing
            last_frame_time = current_time
            frame_count += 1
            
            # Calculate FPS every second
            if current_time - last_fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_fps_time = current_time
                print(f"Current FPS: {fps}")

            # Get new frame
            img = frame_reader.frame
            if img is None:
                print("No frame received from Tello")
                continue
                
            # Only process if frame is different from last frame
            if last_frame is not None and np.array_equal(img, last_frame):
                continue
                
            last_frame = img.copy()
                
            # Run YOLOv8 detection
            img, info = findPerson(img, model)
            
            # Display FPS on frame
            cv2.putText(img, f'FPS: {fps}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Tello Person Detection", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Clean up
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
