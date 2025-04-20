import time
import cv2
from djitellopy import Tello, TelloException
import keyboard
import subprocess
import os

def fix_fps(filename, filename_fixed, fps):
    subprocess.run([
        'ffmpeg', '-y', '-i', filename,
        '-r', str(fps),
        '-c:v', 'copy',
        filename_fixed
    ])
    os.remove(filename)

def main(fly=False, record=False):
    tello = Tello()

    duration = 0
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
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # out = cv2.VideoWriter('drone_output.avi', fourcc, 30.0, (width, height))
        # time.sleep(1)

        frame_count = 0
        start_time = time.time()
        print("Recording started...")

        vel = 20
        while True:
            img = frame_read.frame
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (width, height))
            # out.write(img)
            cv2.imshow("Drone View", img)
            cv2.waitKey(10)
            frame_count += 1

            if fly:
                if keyboard.is_pressed('w'):
                    vx = vel * 3
                elif keyboard.is_pressed('s'):
                    vx = -vel * 4
                else:
                    vx = 0
                if keyboard.is_pressed('r'):
                    vz = vel
                elif keyboard.is_pressed('f'):
                    vz = -vel
                else:
                    vz = 0
                if keyboard.is_pressed('a'):
                    wz = -vel * 3
                elif keyboard.is_pressed('d'):
                    wz = vel * 3
                else:
                    wz = 0

                kp = 0.5
                curr_vx = tello.get_speed_x()
                curr_vy = tello.get_speed_y()
                curr_vz = tello.get_speed_z()
                vx = int(vx + kp * (curr_vx - vx))
                vy = int(0 + kp * (curr_vy - 0))
                vz = int(vz + kp * (curr_vz - vz))
                print(curr_vx, curr_vy, curr_vz)
                tello.send_rc_control(vy, vx, vz, wz)

            if keyboard.is_pressed('q'):
                print("finishing")
                duration = round(time.time() - start_time, 2)
                break
    finally:
        print("Recording duration:", duration, "seconds", frame_count, "frames")
        cv2.destroyAllWindows()
        # out.release()

        if fly:
            try:
                tello.land()
            except:
                pass
        tello.streamoff()
        tello.end()

        if record:
            true_fps = frame_count / duration
            print(f"Fixing video playback to {true_fps:.2f} FPS...")
            fix_fps('drone_output.avi', 'video/drone.avi', true_fps)

if __name__ == "__main__":
    main(fly=True, record=False)
