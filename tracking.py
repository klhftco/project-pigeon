import cv2
from djitellopy import Tello
from collections import deque
import numpy as np
import time
import pickle
from ultralytics import YOLO

# Configuration
TARGET_MIN_SIZE = 40000  # Minimum area in pixels (~1.5m distance)
TARGET_MAX_SIZE = 80000  # Maximum area in pixels (~2m distance)

# --- Drone Setup ---
tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"Tello battery: {battery}%")

if battery < 20:
    print("WARNING: Battery level is low! Landing...")
    tello.land()
    exit()

print("Waiting for drone to stabilize...")
time.sleep(5)

print("Starting video stream...")
tello.streamon()
time.sleep(2)

# --- Takeoff with retry ---
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
        time.sleep(2)

# --- Initial height adjustment ---
tello.send_rc_control(0, 0, 25, 0)
time.sleep(3)

# --- Detection Function ---
def findPerson(img, model):
    results = model(img, classes=[0])  # Class 0 = person
    result = results[0]
    boxes = result.boxes

    myPersonListC = []
    myPersonListArea = []
    person_area_queue.clear()

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        area = (x2 - x1) * (y2 - y1)

        if area < TARGET_MIN_SIZE:
            color = (0, 255, 255)  # Yellow - too far
        elif area > TARGET_MAX_SIZE:
            color = (255, 0, 0)    # Blue - too close
        else:
            color = (0, 255, 0)    # Green - perfect distance

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        conf = box.conf[0].item()
        cv2.putText(img, f'Conf: {conf:.2f}', (x1, y1 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, f'Area: {area}', (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if area < TARGET_MIN_SIZE:
            status = "Too far"
        elif area > TARGET_MAX_SIZE:
            status = "Too close"
        else:
            status = "Good distance"
        cv2.putText(img, status, (x1, y2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myPersonListC.append([cx, cy])
        myPersonListArea.append(area)

    if myPersonListArea:
        i = myPersonListArea.index(max(myPersonListArea))
        return img, [myPersonListC[i], myPersonListArea[i]]
    else:
        return img, [[0, 0], 0]

# --- Queues for smoothing ---
x_error_queue = deque(maxlen=100)
y_error_queue = deque(maxlen=100)
person_coord_queue = deque(maxlen=20)
person_area_queue = deque(maxlen=10)

# --- Tracking Function ---
def trackPerson(info, pid, px_error, py_error, fbRange=[TARGET_MIN_SIZE, TARGET_MAX_SIZE]):
    (x_raw, y_raw), area = info
    person_coord_queue.append((x_raw, y_raw))
    person_area_queue.append(area)

    h, w = 720, 960

    valid_coords = [coord for coord in person_coord_queue if coord != (0, 0)]
    if valid_coords:
        x = int(np.mean([c[0] for c in valid_coords]))
        y = int(np.mean([c[1] for c in valid_coords]))
    else:
        x, y = 0, 0

    valid_areas = [a for a in person_area_queue if a > 0]
    avg_area = int(np.mean(valid_areas)) if valid_areas else 0

    if x == 0 or y == 0:
        x_error = y_error = x_speed = y_speed = fb = z_speed = 0
        # Print search mode status
        print("\n=== SEARCH MODE ===")
        print("No person detected")
        print("Turning slowly to search...")
    else:
        x_error = x - w // 2
        y_error = y - h // 4

        # Print tracking mode status
        print("\n=== TRACKING MODE ===")
        print(f"Person detected at position: ({x}, {y})")
        print(f"Target area: {avg_area} (desired range: {fbRange[0]}-{fbRange[1]})")
        print(f"Position error: ({x_error}, {y_error})")

        x_speed = pid[0] * x_error + pid[1] * (x_error - px_error) + pid[2] * sum(x_error_queue)
        x_speed = int(np.clip(x_speed, -100, 100))

        y_speed = pid[0] * y_error + pid[1] * (y_error - py_error) + pid[2] * sum(y_error_queue)
        y_speed = int(np.clip(-0.5 * y_speed, -40, 40))

        if avg_area > fbRange[1]:
            fb = -20
            print("Moving BACKWARD - Person too close")
        elif avg_area < fbRange[0] and avg_area != 0:
            fb = 20
            print("Moving FORWARD - Person too far")
        else:
            fb = 0
            print("Maintaining distance - Person at ideal range")

        z_speed = -10 if tello.get_height() > 220 else 0
        if z_speed != 0:
            print("Adjusting height - Too high")

    # if no target detected, rotate in place to search
    if x == 0 or y == 0:
        turn_dir = -1
        if sum([1 for x, y in person_coord_queue if x == 0 and y == 0]) > 15:
            print(f"Turning {'left' if turn_dir == -1 else 'right'} at speed {turn_dir * 15}")
            tello.send_rc_control(0, 0, 0, turn_dir * 15)
        else:
            print("Pausing turn - Checking for person")
            tello.send_rc_control(0, 0, 0, 0)
    else:
        print(f"Movement commands: Forward/Back={fb}, Left/Right={x_speed}, Up/Down={z_speed}")
        tello.send_rc_control(0, fb, z_speed, x_speed)

    x_error_queue.append(x_error)
    y_error_queue.append(y_error)
    time.sleep(0.1)

    return x_error, y_error

# --- Main Loop ---
def main():
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    model.predict(np.zeros((720, 960, 3), dtype=np.uint8), classes=[0])  # warm-up

    pid = [0.2, 0.04, 0.005]
    px_error = py_error = 0
    prev_time = time.time()

    print("\n=== DRONE TRACKING STARTED ===")
    print(f"Target size range: {TARGET_MIN_SIZE}-{TARGET_MAX_SIZE} pixels")
    print("Press 'q' to quit\n")

    try:
        while True:
            img = tello.get_frame_read().frame
            img, info = findPerson(img, model)
            px_error, py_error = trackPerson(info, pid, px_error, py_error)

            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()
            cv2.putText(img, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Tello Tracking", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n=== MANUAL LANDING INITIATED ===")
                tello.land()
                break
    except KeyboardInterrupt:
        print("\n=== EMERGENCY STOP ===")
        print("Keyboard interrupt detected")
    finally:
        print("\n=== CLEANUP ===")
        print("Landing drone...")
        tello.land()
        print("Stopping video stream...")
        tello.streamoff()
        print("Closing windows...")
        cv2.destroyAllWindows()
        print("=== DRONE TRACKING ENDED ===\n")

if __name__ == "__main__":
    main()
