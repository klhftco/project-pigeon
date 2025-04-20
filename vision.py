import cv2
from djitellopy import Tello
from collections import deque
import numpy as np
import time
import pickle
from ultralytics import YOLO

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
    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully!")

    tello = Tello()
    tello.connect()
    print(tello.get_battery())
    tello.streamon()

    while True:
        img = tello.get_frame_read().frame
        img, info = findPerson(img, model)
        cv2.imshow("Output", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tello.streamoff()
    tello.end()

if __name__ == "__main__":
    main()
