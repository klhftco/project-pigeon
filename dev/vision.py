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

def findShirt(img):
    '''
    output:
        img - image with bounding box and center overlay
        mask_preview - binary image showing HSV threshold mask
        info[0] - (x, y) coordinate of shirt center in frame
        info[1] - area of the detected shirt region
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV range for deep blue
    lower_blue = np.array([90, 20, 170])
    upper_blue = np.array([105, 100, 255])

    mask = cv2.inRange(imgHSV, lower_blue, upper_blue)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Largest contour = most likely shirt
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2

        if True:
            # Draw box on original image
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        return img, mask, [[cx, cy], area]
    else:
        return img, mask, [[0, 0], 0]

def main():
    tello = Tello()
    tello.connect()
    print(tello.get_battery())
    tello.streamon()
    # cap = cv2.VideoCapture("blue.mp4")
    # avg_area = []

    while True:
        img = tello.get_frame_read().frame
        img, info = findFace(img) # TODO: swap out with YOLO
        cv2.imshow("Output", img)

        # img, mask_preview, info = findShirt(img)
        # mask_preview_bgr = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
        # small_img = cv2.resize(img, (480, 360))
        # small_mask = cv2.resize(mask_preview_bgr, (480, 360))
        # cv2.imshow("Shirt Tracker", small_img)
        # cv2.imshow("Blue Mask", small_mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tello.streamoff()
    tello.end()

if __name__ == "__main__":
    main()
