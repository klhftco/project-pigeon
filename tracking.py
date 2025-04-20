import cv2
from djitellopy import Tello
from collections import deque
import numpy as np
import time
from ultralytics import YOLO

# --- Drone Setup ---
tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"Tello battery: {battery}%")

if battery < 20:
    print("WARNING: Battery level is low! Consider charging the drone.")

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
def findPerson(img, model, conf_threshold=0.5):
    results = model(img, classes=[0])  # Class 0 = person
    result = results[0]
    boxes = result.boxes

    myPersonListC = []
    myPersonListArea = []
    person_area_queue.clear()

    for box in boxes:
        if box.conf[0] < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cx = (x1 + x2) // 2
        cy = int(0.75 * y2 + 0.25 * y1)  # Bias toward lower body
        area = (x2 - x1) * (y2 - y1)

        myPersonListC.append([cx, cy])
        myPersonListArea.append(area)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'Conf: {box.conf[0]:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
def trackPerson(info, pid, px_error, py_error, fbRange=[14000, 15000]):
    (x_raw, y_raw), area = info
    person_coord_queue.append((x_raw, y_raw))
    person_area_queue.append(area)

    w, h = 960, 720

    valid_coords = [coord for coord in person_coord_queue if coord != (0, 0)]
    if valid_coords:
        x = int(np.mean([c[0] for c in valid_coords]))
        y = int(np.mean([c[1] for c in valid_coords]))
    else:
        x, y = 0, 0

    valid_areas = [a for a in person_area_queue if a > 0]
    avg_area = int(np.mean(valid_areas)) if valid_areas else 0

    if x == 0 or y == 0:
        x_error = y_error = x_speed = y_speed = fb = 0
    else:
        x_error = x - w // 2
        y_error = y - h // 4

        x_speed = pid[0] * x_error + pid[1] * (x_error - px_error) + pid[2] * sum(x_error_queue)
        x_speed = int(np.clip(x_speed, -100, 100))

        y_speed = pid[0] * y_error + pid[1] * (y_error - py_error) + pid[2] * sum(y_error_queue)
        y_speed = int(np.clip(-0.5 * y_speed, -40, 40))

        if avg_area > fbRange[1]:
            fb = -15
        elif avg_area < fbRange[0] and avg_area != 0:
            fb = 15
        else:
            fb = 0

        if tello.get_height() > 220:
            tello.send_rc_control(0, 0, -10, 0)

    if x == 0 or y == 0:
        turn_dir = -1 if x < w // 2 else 1
        if sum([1 for x, y in person_coord_queue if x == 0 and y == 0]) > 15:
            tello.send_rc_control(0, 0, 0, turn_dir * 40)
        else:
            tello.send_rc_control(0, 0, 0, 0)
    else:
        tello.send_rc_control(0, fb, y_speed, x_speed)

    x_error_queue.append(x_error)
    y_error_queue.append(y_error)
    time.sleep(0.1)

    print(f"fb={fb} x_speed={x_speed} y_speed={y_speed}")
    return x_error, y_error

# --- Main Loop ---
def main():
    model = YOLO('yolov8n.pt')
    pid = [0.15, 0.03, 0.004]
    px_error = py_error = 0

    try:
        while True:
            img = tello.get_frame_read().frame
            img, info = findPerson(img, model)
            px_error, py_error = trackPerson(info, pid, px_error, py_error)
            cv2.imshow("Tello Tracking", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                tello.land()
                break
    except KeyboardInterrupt:
        print("Interrupted. Landing...")
    finally:
        tello.land()
        tello.streamoff()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
