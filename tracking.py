import cv2
from djitellopy import Tello
from collections import deque
import numpy as np
import time
import pickle
from ultralytics import YOLO

tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"Tello battery: {battery}%")

# Check battery level but only warn
if battery < 20:
    print("WARNING: Battery level is low! Consider charging the drone.")

# Wait for drone to stabilize
print("Waiting for drone to stabilize...")
time.sleep(5)

# Start video stream
print("Starting video stream...")
tello.streamon()
time.sleep(2)  # Give time for stream to initialize

# Try to take off with retries
max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"Takeoff attempt {attempt + 1}...")
        tello.takeoff()
        print("Takeoff successful!")
        break
    except Exception as e:
        print(f"Takeoff attempt {attempt + 1} failed: {str(e)}")
        if attempt == max_retries - 1:
            print("All takeoff attempts failed. Please check the drone and try again.")
            tello.streamoff()
            exit()
        time.sleep(2)  # Wait before retrying

# Initial height adjustment
tello.send_rc_control(0, 0, 25, 0)
time.sleep(3)

def findPerson(img, model):
    '''
    input:
        img - RGB image as an np.array
        model - YOLOv8 model
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

# initialize center coord, error queues
x_error_queue = deque(maxlen=100)
y_error_queue = deque(maxlen=100)
person_coord_queue = deque(maxlen=20)

def trackPerson(info, pid, px_error, py_error, fbRange=[14000, 15000]):
    (x, y), area = info
    w, h = 960, 720
    person_coord_queue.append((x, y))

    # reset errors if no person is detected
    if x == 0 or y == 0:
        x_error = 0
        y_error = 0
        x_speed = 0
        y_speed = 0
        fb = 0
    else:
        x_error = x - w // 2
        y_error = y - h // 4

        print(x, y)
        print(x_error, y_error)

        # calculate vx, vy based on pid controller, pseudo-integration
        x_speed = pid[0] * x_error + pid[1] * (x_error - px_error) + pid[2] * sum(x_error_queue)
        x_speed = int(np.clip(x_speed, -100, 100))

        y_speed = pid[0] * y_error + pid[1] * (y_error - py_error) + pid[2] * sum(y_error_queue)
        y_speed = int(np.clip(-0.5 * y_speed, -40, 40))

        # calculate forward/backward speed based on area of target
        if area > fbRange[1]:
            fb = -20 # backward
        elif area < fbRange[0] and area != 0:
            fb = 20 # forward
        else:
            fb = 0 # stop

        # limit height of drone to prevent ceiling collision
        if tello.get_height() > 220:
            tello.send_rc_control(0, 0, -10, 0)

    # if no target detected, rotate in place to search
    if x == 0 or y == 0:
        turn_dir = -1 if x < w // 2 else 1
        if sum([1 for x, y in person_coord_queue if x == 0 and y == 0]) > 15:
            tello.send_rc_control(0, 0, 0, turn_dir * 40)
        else:
            tello.send_rc_control(0, 0, 0, 0)
    else:
        # otherwise, send RC control to follow target
        tello.send_rc_control(0, fb, y_speed, x_speed)

    x_error_queue.append(x_error)
    y_error_queue.append(y_error)
    time.sleep(0.1)

    print(f"fb={fb} x_speed={x_speed} y_speed={y_speed}")
    return x_error, y_error

def main():
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # pid parameters
    pid = [0.2, 0.04, 0.005]
    px_error = 0
    py_error = 0

    try:
        while True:
            img = tello.get_frame_read().frame
            img, info = findPerson(img, model)
            px_error, py_error = trackPerson(info, pid, px_error, py_error)
            cv2.imshow("Output", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                tello.land()
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Clean up
        print("Landing...")
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
