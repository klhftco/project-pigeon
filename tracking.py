import cv2
from djitellopy import Tello
from collections import deque
import numpy as np
import time
import pickle

tello = Tello()
tello.connect()
print(tello.get_battery())

tello.streamon()
tello.takeoff()
tello.send_rc_control(0, 0, 25, 0)
time.sleep(3)

def findFace(img):
    '''
    output:
        img - image overlay of tracking target
        info[0] - (x, y) coordinate of center of target in frame
        info[1] - area of target in frame
    '''
    # return the image with bounding box around the face & the center of the face
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []
    for x, y, w, h in faces:
        # draw bounding box around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:
        # return the face with the largest area
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

# initialize center coord, error queues
x_error_queue = deque(maxlen=100)
y_error_queue = deque(maxlen=100)
face_coord_queue = deque(maxlen=20)

def trackFace(info, pid, px_error, py_error, fbRange=[14000, 15000]):
    (x, y), area = info
    w, h = 960, 720
    face_coord_queue.append((x, y))

    # reset errors if no face is detected
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
        if sum([1 for x, y in face_coord_queue if x == 0 and y == 0]) > 15:
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
    # pid parameters
    pid = [0.2, 0.04, 0.005]
    px_error = 0
    py_error = 0

    while True:
        img = tello.get_frame_read().frame
        img, info = findFace(img) # TODO: swap out with YOLO
        px_error, py_error = trackFace(info, pid, px_error, py_error)
        cv2.imshow("Output", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            tello.land()
            break

if __name__ == "__main__":
    main()
