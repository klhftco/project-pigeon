import time
import cv2
from djitellopy import Tello, TelloException
import keyboard


def main(fly=False):
    tello = Tello()

    try:
        tello.connect()
        print(tello.get_battery())
        tello.streamon()
        if fly:
            tello.takeoff()

        frame_read = tello.get_frame_read()
        frame = frame_read.frame
        height, width, _ = frame.shape

        # Use a codec known to work well on Windows
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('drone_output.avi', fourcc, 80.0, (width, height))
        time.sleep(1)

        start_time = time.time()
        print("Recording started...")

        vel = 20
        while True:
            img = frame_read.frame
            img = cv2.resize(img, (width, height))
            out.write(img)
            cv2.imshow("Drone View", img)
            cv2.waitKey(10)

            if fly:
                if keyboard.is_pressed('w'):
                    vx = vel * 2
                    wz = 0
                elif keyboard.is_pressed('s'):
                    vx = -vel * 3
                    wz = 0
                elif keyboard.is_pressed('a'):
                    vx = 0
                    wz = -vel * 3
                elif keyboard.is_pressed('d'):
                    vx = 0
                    wz = vel * 3
                else:
                    vx = 0
                    wz = 0
                tello.send_rc_control(0, vx, 0, wz)

            if keyboard.is_pressed('q'):
                print("finishing")
                break
    finally:
        print("Recording duration:", round(time.time() - start_time, 2), "seconds")
        cv2.destroyAllWindows()
        out.release()

        if fly:
            try:
                tello.land()
            except:
                pass
        tello.streamoff()
        tello.end()

if __name__ == "__main__":
    main(fly=False)
