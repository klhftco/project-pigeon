from motion_controller import MotionController
import time
import traceback
import cv2
import numpy as np
import threading
from collections import deque
from queue import Queue, Empty
from djitellopy import Tello
from ultralytics import YOLO

class DroneController:
    def __init__(self, config):
        self.config = config
        self.tello = Tello()
        self.model = YOLO(config["yolo_model"])
        self.state = "SEARCHING"
        self.lost_counter = 0
        self.confirm_counter = 0
        self.px_error = 0
        self.py_error = 0
        self.x_error_queue = deque(maxlen=100)
        self.y_error_queue = deque(maxlen=100)
        self.frame_queue = Queue(maxsize=3)
        self.result_queue = Queue(maxsize=3)
        self.stop_threads = threading.Event()
        self.motion_controller = MotionController()
        self.is_rotating_360 = False
        self.rotation_start_time = None
        self.rotation_duration = 3.0  # seconds for full 360-degree rotation

    def start(self):
        self.motion_controller.start()

    def stop(self):
        self.motion_controller.stop()

    def _control_loop(self):
        while self._running:
            try:
                # Check for 360-degree motion request
                if self.motion_controller.check_360_motion_requested() and not self.is_rotating_360:
                    print("Starting 360-degree rotation")
                    self.is_rotating_360 = True
                    self.rotation_start_time = time.time()
                    self.tello.rotate_clockwise(360)  # Start rotation
                
                # Check if rotation is complete
                if self.is_rotating_360:
                    elapsed_time = time.time() - self.rotation_start_time
                    if elapsed_time >= self.rotation_duration:
                        print("360-degree rotation complete")
                        self.is_rotating_360 = False
                        self.rotation_start_time = None
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                traceback.print_exc()
                continue 

    def run(self):
        frame_thread = threading.Thread(target=self.frame_grabber)
        infer_thread = threading.Thread(target=self.model_inference)
        frame_thread.start()
        infer_thread.start()
        self.motion_controller.start()  # Start motion controller

        # Create display window
        cv2.namedWindow("Tello Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tello Tracking", 960, 720)

        try:
            while True:
                try:
                    img, info = self.result_queue.get(timeout=0.1)
                    
                    # Check for keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    self.motion_controller.handle_key(key)
                    
                    # Check for 360-degree motion request
                    if self.motion_controller.check_360_motion_requested() and not self.is_rotating_360:
                        print("Starting 360-degree rotation")
                        self.is_rotating_360 = True
                        self.rotation_start_time = time.time()
                        self.tello.rotate_clockwise(360)  # Start rotation
                    
                    # Check if rotation is complete
                    if self.is_rotating_360:
                        elapsed_time = time.time() - self.rotation_start_time
                        if elapsed_time >= self.rotation_duration:
                            print("360-degree rotation complete")
                            self.is_rotating_360 = False
                            self.rotation_start_time = None
                            if self.state == "TRACKING":
                                self.state = "SEARCHING"  # Reset to searching after rotation
                    
                    # Display the frame
                    cv2.imshow("Tello Tracking", img)
                    
                except Empty:
                    continue
                except Exception as e:
                    print(f"Error in main loop: {str(e)}")
                    traceback.print_exc()
                    break

        except KeyboardInterrupt:
            print("Keyboard interrupt detected, landing drone...")
        except Exception as e:
            print(f"Fatal error: {str(e)}")
            traceback.print_exc()
        finally:
            self.stop_threads.set()
            self.motion_controller.stop()  # Stop motion controller
            frame_thread.join()
            infer_thread.join()
            self.tello.land()
            self.tello.streamoff()
            cv2.destroyAllWindows() 