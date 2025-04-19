import cv2
import numpy as np
from djitellopy import Tello
import time
from collections import deque
import pickle

tello = Tello()
tello.connect()
print(tello.get_battery())

tello.streamon()
tello.takeoff()
tello.send_rc_control(0, 0, 25, 0)
time.sleep(3)

#resize the frame
w, h = 960, 720
#face detection range
fbRange = [14000, 15000]
#pid parameters
pid = [0.2, 0.04, 0.005]
INDEX = 0

def findFace(img):
    #return the image with bounding box around the face & the center of the face
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []
    for x, y, w, h in faces:
        #draw bounding box around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:
        #return the face with the largest area
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

#initialize the error queues
x_error_queue = deque(maxlen=100)
y_error_queue = deque(maxlen=100)
#initialize the center coordinates queue, which stores the last 20 center coordinates
center_coords_queue = deque(maxlen=20)

#default turn direction
TURN = ["+"]
def trackFace(info, w, pid, px_error, py_error):
    area = info[1]
    x, y = info[0]

    center_coords_queue.append((x, y))
    #determine the turn direction based on the x coordinate of the face.
    if x != 0 or y != 0:
        if x < w // 2:
            TURN[0] = "-"
        else:
            TURN[0] = "+"
    x_error = x - w // 2
    y_error = y - h // 4

    if x == 0 or y == 0:
        x_error = 0
        y_error = 0

    print(x,y)
    print(x_error, y_error)
    #calculate the x and y speeds based on the PID controller
    x_speed = (
        pid[0] * x_error + pid[1] * (x_error - px_error) + pid[2] * sum(x_error_queue)
    )

    x_speed = int(np.clip(x_speed, -100, 100))

    y_speed = (
        pid[0] * y_error + pid[1] * (y_error - py_error) + pid[2] * sum(y_error_queue)
    )
    y_speed = int(np.clip(-0.5 * y_speed, -40, 40))

    #calculate the forward/backward speed based on the area of the face
    if area >= fbRange[0] and area <= fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    else:
        fb = 0

    #since our demo is done indoor, we want to limit the height of the drone so it won't hit the ceiling
    if tello.get_height() > 220:
        tello.send_rc_control(0, 0, -10, 0)

    #if the face is not detected, we want the drone to rotate in place to search for the face
    if x == 0 or y == 0:
        if sum([1 for x, y in center_coords_queue if x == 0 and y == 0]) > 15:
            if TURN[0] == "+":
                tello.send_rc_control(0, 0, 0, 40)
            else:
                tello.send_rc_control(0, 0, 0, -40)
        else:
            tello.send_rc_control(0, 0, 0, 0)
        x_error = 0
        y_error = 0
        x_speed = 0
        y_speed = 0
    else:
        #Otherwise, we send the RC control commands to follow the face
        tello.send_rc_control(0, fb, y_speed, x_speed)

    x_error_queue.append(x_error)
    y_error_queue.append(y_error)

    time.sleep(0.1)

    print(f"fb={fb} x_speed={x_speed} y_speed={y_speed}")
    return x_error, y_error

def circle_motion(drone, speed, yaw_speed, duration):
    # record a panoramic video by moving the drone in a circular motion
    print(f"Starting circular motion with speed={speed} cm/s, yaw_speed={yaw_speed} deg/s, and duration={duration} seconds")

    start_time = time.time()

    height, width, _ = drone.get_frame_read().frame.shape
    video = cv2.VideoWriter(f'circle_motion_video_{start_time}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    print("Video recording started.")

    try:
        while time.time() - start_time < duration:
            frame = drone.get_frame_read().frame
            cv2.imshow("Output", frame)

            if frame is not None:
                video.write(frame)
            time.sleep(1 / 50)

            # Check if "q" is pressed to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[DEBUG] 'q' pressed, stopping and landing...")
                break

            # Send RC control commands: move forward and yaw
            drone.send_rc_control(speed, 0, 0, yaw_speed)
            time.sleep(0.05)  # Maintain smooth control by sending commands frequently

    except Exception as e:
        print(f"[ERROR] An error occurred during circular motion: {e}")

    finally:
        # Stop the motion
        print("[DEBUG] Stopping drone...")
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(0.1)
        video.release()
        print("Video recording stopped and saved as 'circle_motion_video.mp4'.")
        
    # # Example usage of circle_motion in main:
    # circle_motion(me, speed=-30, yaw_speed=-35, duration=24)  # Adjust speed, yaw speed, and duration as needed

def take_picture(img):
    #Save a picture from the current frame + bounding box.
    cv2.imwrite(f"picture_{time.time()}.png", img)
    print("Picture taken!")

def read_sensor_and_perform_action(img):
    # time.sleep(2)
    # print("command sent!!!!")
    # execute_command(1, img)
    with open("cache.pkl", "rb") as file:
        allSnaps = pickle.load(file)
    global INDEX
    if len(allSnaps)>INDEX:
        print("current command:", allSnaps[INDEX])
        Command = allSnaps[INDEX]
        INDEX +=1
        execute_command(Command, img)

def execute_command(command, img):
    print("executing command:", command)
    if command == 1:
        print("Yield to circle motion")
        circle_motion(me, speed=-15, yaw_speed=35, duration=20)
        print("Circle motion done")
    elif command == 2:
        take_picture(img)
    else:
        pass

px_error = 0
py_error = 0

# Main Control Loop
while True:
    img = tello.get_frame_read().frame
    img, info = findFace(img)
    px_error, py_error = trackFace(info, w, pid, px_error, py_error)
    read_sensor_and_perform_action(img)
    cv2.imshow("Output", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        tello.land()
        break

# import matplotlib.pyplot as plt

# plt.hist(p_ls, label="p")
# plt.hist(i_ls, label="i")
# plt.hist(d_ls, label="d")

# print(p_ls[-100:], i_ls[-100:], d_ls[-100:])
# plt.legend()
# plt.show()