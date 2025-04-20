import cv2
from djitellopy import Tello
from collections import deque
import numpy as np
import time
import pickle

def findFace(img):
    '''
    output:
        img - image overlay of tracking target
        info[0] - (x, y) coordinate of center of target in frame
        info[1] - area of target in frame
    '''
    # return the image with bounding box around the face & the center of the face
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []
    for x, y, w, h in faces:
        # draw bounding box around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
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

def main():
    tello = Tello()
    tello.connect()
    print(tello.get_battery())
    tello.streamon()

    while True:
        img = tello.get_frame_read().frame
        img, info = findFace(img) # TODO: swap out with YOLO
        cv2.imshow("Output", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tello.streamoff()
    tello.end()

if __name__ == "__main__":
    main()
