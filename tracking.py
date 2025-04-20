import cv2
import time
import numpy as np
import threading
import subprocess
from collections import deque
from queue import Queue, Empty
from djitellopy import Tello
from ultralytics import YOLO
import argparse
from datetime import datetime

class VideoStream:
    def __init__(self, max_queue_size=3):
        self.tello = Tello()
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.stream_thread = None
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.width, self.height = 1280, 720  # Tello video resolution
        self.frame_size = self.width * self.height * 3  # 3 bytes per pixel (BGR24)
        self.ffmpeg_process = None

    def start_stream(self):
        """Start the video stream thread immediately"""
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        return self

    def _start_ffmpeg(self):
        """Start FFmpeg process to capture video stream"""
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", "udp://0.0.0.0:11111",  # Tello's default video stream address
            "-pix_fmt", "bgr24",           # pixel format for OpenCV (BGR)
            "-f", "rawvideo",
            "-"
        ]
        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**7
        )

    def _stream_loop(self):
        """Main streaming loop using FFmpeg"""
        try:
            self._start_ffmpeg()
            while not self.stop_event.is_set():
                try:
                    # Read raw frame from FFmpeg
                    raw_frame = self.ffmpeg_process.stdout.read(self.frame_size)
                    if not raw_frame:
                        print("FFmpeg stream ended, restarting...")
                        self._start_ffmpeg()
                        continue

                    # Convert raw frame to numpy array
                    frame = np.frombuffer(raw_frame, np.uint8).reshape((self.height, self.width, 3))

                    # Calculate FPS
                    self.frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_frame_time >= 1.0:
                        self.fps = self.frame_count
                        self.frame_count = 0
                        self.last_frame_time = current_time

                    # Put frame in queue
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)

                except Exception as e:
                    print(f"Stream error: {e}")
                    time.sleep(0.1)  # Brief pause before retrying

        finally:
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()

    def get_frame(self, timeout=0.1):
        """Get the latest frame from the queue"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_fps(self):
        """Get current FPS"""
        return self.fps

    def stop(self):
        """Stop the video stream"""
        self.stop_event.set()
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
        if self.stream_thread:
            self.stream_thread.join()

class FrameProcessor:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.processed_queue = Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.process_thread = None
        self.frame_count = 0
        self.processing_time = 0
        self.last_time = time.time()

    def start_processing(self, frame_queue):
        """Start the processing thread"""
        self.process_thread = threading.Thread(target=self._process_loop, args=(frame_queue,))
        self.process_thread.daemon = True
        self.process_thread.start()
        return self

    def _process_loop(self, frame_queue):
        """Main processing loop"""
        while not self.stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
                if frame is not None:
                    start_time = time.time()
                    processed_frame, info = self.process_frame(frame)
                    self.processing_time = time.time() - start_time
                    
                    if self.processed_queue.full():
                        self.processed_queue.get_nowait()
                    self.processed_queue.put_nowait((processed_frame, info))
            except Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def process_frame(self, frame):
        """Process a single frame with YOLO"""
        try:
            results = self.model(frame, classes=[0])[0]
            boxes = results.boxes

            candidates = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                area = (x2 - x1) * (y2 - y1)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                candidates.append(((cx, cy), area, box.conf[0].item(), (x1, y1, x2, y2)))

            if not candidates:
                return frame, [[0, 0], 0]

            best = max(candidates, key=lambda c: c[1])
            (cx, cy), area, conf, (x1, y1, x2, y2) = best

            color = (0, 255, 255) if area < self.config["TARGET_MIN_SIZE"] else (255, 0, 0) if area > self.config["TARGET_MAX_SIZE"] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'Conf: {conf:.2f}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f'Area: {area}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, "Good distance" if color == (0, 255, 0) else ("Too far" if color == (0, 255, 255) else "Too close"),
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            return frame, [[cx, cy], area]
        except Exception as e:
            print(f"Processing error: {e}")
            return frame, [[0, 0], 0]

    def get_processed_frame(self, timeout=0.1):
        """Get the latest processed frame"""
        try:
            return self.processed_queue.get(timeout=timeout)
        except Empty:
            return None, None

    def get_processing_stats(self):
        """Get processing statistics"""
        return {
            "processing_time": self.processing_time,
            "fps": 1.0 / self.processing_time if self.processing_time > 0 else 0
        }

    def stop(self):
        """Stop the processing thread"""
        self.stop_event.set()
        if self.process_thread:
            self.process_thread.join()

class DroneController:
    def __init__(self, config):
        self.config = config
        self.video_stream = VideoStream()
        self.frame_processor = FrameProcessor(YOLO(config["yolo_model"]), config)
        self.state = "SEARCHING"
        self.lost_counter = 0
        self.confirm_counter = 0
        self.px_error = 0
        self.py_error = 0
        self.x_error_queue = deque(maxlen=100)
        self.y_error_queue = deque(maxlen=100)
        self.stop_event = threading.Event()
        self.last_command_time = time.time()
        self.command_interval = 0.1  # Minimum time between commands

    def connect_and_prepare(self):
        try:
            # Start video stream immediately
            self.video_stream.start_stream()
            
            # Connect to drone
            print("Connecting to Tello...")
            self.video_stream.tello.connect()
            battery = self.video_stream.tello.get_battery()
            print(f"Battery: {battery}%")
            if battery < 20:
                print("WARNING: Low battery. Please charge.")

            # Initialize stream
            print("Initializing video stream...")
            self.video_stream.tello.streamon()
            time.sleep(1)  # Reduced from 2 to 1 second

            # Start frame processing
            self.frame_processor.start_processing(self.video_stream.frame_queue)

            print("Taking off...")
            self.video_stream.tello.takeoff()
            time.sleep(1)  # Reduced from 2 to 1 second

            print("Ascending to ~1.8m height...")
            start_time = time.time()
            while self.video_stream.tello.get_height() < 180 and time.time() - start_time < 10:  # Added timeout
                self.video_stream.tello.send_rc_control(0, 0, 25, 0)
                time.sleep(0.05)  # Reduced from 0.1 to 0.05
            self.video_stream.tello.send_rc_control(0, 0, 0, 0)
            self.video_stream.tello.send_rc_control(0, 0, 0, 15)

        except Exception as e:
            print(f"Initialization error: {e}")
            self.emergency_land()
            raise

    def emergency_land(self):
        """Emergency landing procedure"""
        try:
            print("EMERGENCY LANDING!")
            self.video_stream.tello.land()
            self.video_stream.tello.streamoff()
        except:
            pass

    def send_control(self, lr, fb, ud, yaw):
        """Send control commands with rate limiting"""
        current_time = time.time()
        if current_time - self.last_command_time >= self.command_interval:
            try:
                self.video_stream.tello.send_rc_control(lr, fb, ud, yaw)
                self.last_command_time = current_time
            except Exception as e:
                print(f"Control error: {e}")

    def track_person(self, info):
    (x, y), area = info
    w, h = 960, 720

    if x == 0 or y == 0:
            self.send_control(0, 0, 0, 0)
            return

        x_error = x - w // 2
        y_error = y - h // 4

        x_speed = self.config["pid"][0] * x_error + self.config["pid"][1] * (x_error - self.px_error) + self.config["pid"][2] * sum(self.x_error_queue)
        y_speed = self.config["pid"][0] * y_error + self.config["pid"][1] * (y_error - self.py_error) + self.config["pid"][2] * sum(self.y_error_queue)

        x_speed = int(np.clip(x_speed, -100, 100))
        y_speed = int(np.clip(-0.5 * y_speed, -40, 40))

        fb = -20 if area > self.config["TARGET_MAX_SIZE"] else 20 if area < self.config["TARGET_MIN_SIZE"] else 0
        z_speed = -10 if self.video_stream.tello.get_height() > 220 else 0

        self.send_control(0, fb, z_speed, x_speed)
        self.px_error = x_error
        self.py_error = y_error
        self.x_error_queue.append(x_error)
        self.y_error_queue.append(y_error)

    def run(self):
        try:
            print("\n=== DRONE TRACKING STARTED ===")
            print("Press 'q' to quit\n")

            while not self.stop_event.is_set():
                processed_frame, info = self.frame_processor.get_processed_frame()
                if processed_frame is None:
                    continue

                # Add FPS and processing time to frame
                stream_fps = self.video_stream.get_fps()
                proc_stats = self.frame_processor.get_processing_stats()
                cv2.putText(processed_frame, f'Stream FPS: {stream_fps}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(processed_frame, f'Proc FPS: {proc_stats["fps"]:.1f}', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(processed_frame, f'Proc Time: {proc_stats["processing_time"]*1000:.1f}ms', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                area = info[1]
                if self.state == "SEARCHING":
                    if self.config["TARGET_MIN_SIZE"] < area < self.config["TARGET_MAX_SIZE"]:
                        self.confirm_counter += 1
                        print(f"Confirming target... ({self.confirm_counter}/{self.config['CONFIRMATION_FRAMES']})")
                        if self.confirm_counter >= self.config["CONFIRMATION_FRAMES"]:
                            print("Target confirmed. Switching to TRACKING.")
                            self.send_control(0, 0, 0, 0)
                            self.state = "TRACKING"
                            self.lost_counter = 0
                            self.confirm_counter = 0
        else:
                        self.confirm_counter = 0
                elif self.state == "TRACKING":
                    if area == 0:
                        self.lost_counter += 1
                        print(f"Lost target ({self.lost_counter}/{self.config['MAX_LOST_FRAMES']})")
                        if self.lost_counter >= self.config["MAX_LOST_FRAMES"]:
                            print("Target lost. Returning to SEARCHING mode.")
                            self.send_control(0, 0, 0, 15)
                            self.state = "SEARCHING"
    else:
                        self.lost_counter = 0
                        self.track_person(info)

                cv2.imshow("Tello Tracking", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        except KeyboardInterrupt:
            print("\n=== EMERGENCY STOP ===")
        except Exception as e:
            print(f"\n=== ERROR: {str(e)} ===")
        finally:
            self.stop_event.set()
            self.frame_processor.stop()
            self.video_stream.stop()
            self.emergency_land()
            cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description='Tello Drone Person Tracking')
    parser.add_argument('--min-size', type=int, default=40000,
                        help='Minimum target size in pixels')
    parser.add_argument('--max-size', type=int, default=80000,
                        help='Maximum target size in pixels')
    parser.add_argument('--lost-frames', type=int, default=10,
                        help='Number of frames to tolerate losing sight')
    parser.add_argument('--confirm-frames', type=int, default=5,
                        help='Number of frames required to confirm target')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model to use')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    config = {
        "TARGET_MIN_SIZE": args.min_size,
        "TARGET_MAX_SIZE": args.max_size,
        "MAX_LOST_FRAMES": args.lost_frames,
        "CONFIRMATION_FRAMES": args.confirm_frames,
        "yolo_model": args.model,
        "pid": [0.2, 0.04, 0.005]
    }

    controller = DroneController(config)
    controller.connect_and_prepare()
    controller.run()
