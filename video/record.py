import time
import cv2
from djitellopy import Tello

tello = Tello()
tello.connect()
print(tello.get_battery())
tello.streamon()
time.sleep(3)

frame_read = tello.get_frame_read()
frame = frame_read.frame
height, width, _ = frame.shape

# Use a codec known to work well on Windows
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('drone_output.avi', fourcc, 80.0, (width, height))

start_time = time.time()
print("Recording started...")

while True:
    img = frame_read.frame
    img = cv2.resize(img, (width, height))

    out.write(img)
    cv2.imshow("Drone View", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Recording duration:", round(time.time() - start_time, 2), "seconds")

cv2.destroyAllWindows()
out.release()
tello.streamoff()
tello.end()
