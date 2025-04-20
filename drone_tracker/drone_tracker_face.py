#!/usr/bin/env python3
import time
import cv2
import numpy as np
import os
from collections import deque
from djitellopy import Tello
import logging

# import your existing modules
from .yolo_detector import YoloDetector
from .tracker import SimpleTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------
# Face‐tracking utils (adapted from face_tracker.py)
# --------------------
# Setup proper path for the cascade file
cascade_file = "haarcascade_frontalface_default.xml"
# Look in multiple locations - current dir, opencv's data dir, and common locations
cascade_paths = [
    cascade_file,
    os.path.join(cv2.__path__[0], 'data', cascade_file),
    os.path.join('drone_tracker', cascade_file),
    os.path.join(os.path.dirname(__file__), cascade_file)
]

face_cascade = None
for path in cascade_paths:
    if os.path.exists(path):
        face_cascade = cv2.CascadeClassifier(path)
        print(f"✅ Found face cascade file at: {path}")
        break

if face_cascade is None or face_cascade.empty():
    print("⚠️ Face cascade file not found or invalid, downloading...")
    import urllib.request
    url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{cascade_file}"
    try:
        urllib.request.urlretrieve(url, cascade_file)
        face_cascade = cv2.CascadeClassifier(cascade_file)
        if face_cascade.empty():
            print("⚠️ Downloaded cascade file is invalid")
        else:
            print(f"✅ Downloaded face cascade to: {os.path.abspath(cascade_file)}")
    except Exception as e:
        print(f"❌ Failed to download cascade file: {e}")
        print("⚠️ Will continue without face detection")

# PID queues
x_error_queue = deque(maxlen=100)
y_error_queue = deque(maxlen=100)
face_coord_queue = deque(maxlen=20)

# Face tracking thresholds
FACE_LOCK_THRESHOLD = 10000  # Minimum face area to switch to face-only tracking
FACE_LOST_COUNT = 10         # Frames without face detection before reverting to YOLO

def find_face(img):
    """Detect largest face in BGR image. Returns (cx, cy), area and draws rect."""
    if face_cascade is None or face_cascade.empty():
        h, w = img.shape[:2]
        return ([w//2, h//2], 0)  # Return center coords with zero area

    try:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(imgGray, 1.2, 8)

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
            return (myFaceListC[i], myFaceListArea[i])
        else:
            return ([0, 0], 0)
    except Exception as e:
        print(f"⚠️ Error in face detection: {e}")
        h, w = img.shape[:2]
        return ([0, 0], 0)

def track_face(tello, info, pid, prev_error, fbRange=[14000, 15000]):
    """
    info: ((cx,cy), area) in the *cropped* ROI
    prev_error: (px_error, py_error)
    pid: [Kp, Ki, Kd]
    Returns new errors and sends rc_control internally.
    """
    try:
        (x, y), area = info

        # If no face is detected (coordinates are zeros), rotate to search
        if x == 0 or y == 0 or area == 0:
            face_coord_queue.append((x, y))
            turn_dir = 1  # Default turn direction

            # If we've had no detections for a while, search by rotating
            if sum([1 for cx, cy in face_coord_queue if cx == 0 and cy == 0]) > 15:
                print("No face detected for a while, searching...")
                tello.send_rc_control(0, 0, 0, turn_dir * 20)
            else:
                # Just hover if we temporarily lost the face
                tello.send_rc_control(0, 0, 0, 0)

            return (0, 0)

        # We have a face, calculate PID control
        x_error = x
        y_error = y

        # Calculate PID values
        x_error_queue.append(x_error)
        y_error_queue.append(y_error)

        # P term
        # D term
        de = x_error - prev_error[0]
        # I term
        ie = sum(x_error_queue)

        # Calculate speed from PID formula
        x_speed = pid[0] * x_error + pid[1] * de + pid[2] * ie
        x_speed = int(np.clip(x_speed, -100, 100))

        # Y-axis PID
        de_y = y_error - prev_error[1]
        ie_y = sum(y_error_queue)
        y_speed = pid[0] * y_error + pid[1] * de_y + pid[2] * ie_y
        y_speed = int(np.clip(-0.5 * y_speed, -40, 40))

        # Forward/backward based on face area
        fb = 0
        if area > fbRange[1]:
            fb = -20  # backward
        elif area < fbRange[0] and area != 0:
            fb = 20   # forward

        print(f"tracking face - x_err: {x_error}, y_err: {y_error}, area: {area}, fb: {fb}, x_speed: {x_speed}, y_speed: {y_speed}")
        tello.send_rc_control(0, fb, y_speed, x_speed)

        return (x_error, y_error)
    except Exception as e:
        print(f"❌ Error in track_face: {e}")
        tello.send_rc_control(0, 0, 0, 0)  # Hover safely on error
        return prev_error  # Return previous error to avoid sudden changes

def expand_roi(x1, y1, x2, y2, frame_width, frame_height, expansion_factor=1.5):
    """Expand ROI by expansion_factor while keeping within frame boundaries."""
    width = x2 - x1
    height = y2 - y1

    # Calculate center
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Calculate new dimensions
    new_width = int(width * expansion_factor)
    new_height = int(height * expansion_factor)

    # Calculate new corners
    new_x1 = max(0, center_x - new_width // 2)
    new_y1 = max(0, center_y - new_height // 2)
    new_x2 = min(frame_width, center_x + new_width // 2)
    new_y2 = min(frame_height, center_y + new_height // 2)

    return new_x1, new_y1, new_x2, new_y2

# --------------------
# Combined Drone‐Face‐Tracker
# --------------------
class DroneFaceTracker:
    def __init__(self, tello=None):
        # drone + vision
        self.tello = tello
        if not tello:
            self.tello = Tello()
        self.detector = YoloDetector(model_path="yolov8n.pt")
        self.tracker = SimpleTracker(iou_threshold=0.3, max_age=10, min_hits=3)
        # PID params for face
        self.pid = [0.2, 0.04, 0.005]
        self.prev_error = (0,0)
        self.is_flying = False
        self.fbRange = [14000, 15000]  # Forward/backward range based on face area
        self.window_name = "Drone View"

        # FPS calculation
        self.last_fps_time = time.time()
        self.frame_count_fps = 0
        self.display_fps = 0.0

        # Battery tracking
        self.battery_level = None
        self.last_battery_check_time = 0
        self.battery_check_interval = 5  # seconds

        # UI settings
        self.horizontal_threshold = 15.0  # Target % offset threshold for turning

        # Face tracking mode
        self.face_mode = False
        self.face_roi = None  # Current region of interest for face tracking
        self.last_face_info = None
        self.face_lost_counter = 0

    def setup(self):
        try:
            self.tello.connect()
            self.battery_level = self.tello.get_battery()
            self.last_battery_check_time = time.time()
            print(f"🔋 Battery: {self.battery_level}%")

            if self.battery_level < 15:
                print("⚠️ Battery critically low! Please charge before flying.")
                return False

            self.tello.streamon()
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.tracker.handle_mouse_click)
            time.sleep(2)
            return True
        except Exception as e:
            print(f"❌ Setup error: {e}")
            return False

    def _draw_overlay(self, frame, confirmed_tracks, target_relative_pos=None):
        """Draw tracking boxes, face info, status and help text."""
        height, width = frame.shape[:2]

        # Draw face tracking mode indicator
        mode_text = "Mode: FACE TRACKING" if self.face_mode else "Mode: PERSON TRACKING"
        mode_color = (0, 255, 0) if self.face_mode else (255, 255, 0)
        cv2.putText(frame, mode_text, (10, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # Draw ROI if in face mode
        if self.face_mode and self.face_roi is not None:
            x1, y1, x2, y2 = self.face_roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Purple for face ROI

        # In person tracking mode, draw all confirmed tracks
        if not self.face_mode:
            for track in confirmed_tracks:
                box = list(map(int, track['box']))
                track_id = track['id']
                x0, y0, x1, y1 = box
                # Basic color variation
                color = ((track_id * 30) % 255, (track_id * 50) % 255, (track_id * 70) % 255)
                thickness = 2
                label = f"ID:{track_id}"

                # Highlight selected (manual click) and target tracks
                is_selected = track_id == self.tracker.selected_track_id
                is_target = track_id == self.tracker.target_track_id

                if is_selected:
                    color = (0, 255, 255)  # Bright Yellow
                    thickness = 3
                    label += " [SELECTED]"
                elif is_target:
                    color = (255, 255, 0)  # Bright Cyan
                    thickness = 3
                    label += " [TARGET]"

                cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
                cv2.putText(frame, label, (x0, y0 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

        # Draw Center Line and Threshold Lines
        center_x = width // 2
        threshold_px = int(center_x * (self.horizontal_threshold / 100.0))
        cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 255), 1)  # Center line (Red)
        cv2.line(frame, (center_x - threshold_px, 0),
                 (center_x - threshold_px, height), (255, 0, 0), 1)  # Left Threshold (Blue)
        cv2.line(frame, (center_x + threshold_px, 0),
                 (center_x + threshold_px, height), (255, 0, 0), 1)  # Right Threshold (Blue)

        # Display Target Info
        if self.face_mode and self.last_face_info and self.last_face_info[1] > 0:
            (cx, cy), area = self.last_face_info
            rel_x = ((cx - width/2) / (width/2)) * 100  # Convert to percentage
            cv2.putText(frame, f"Face Offset: {rel_x:.1f}% Area: {area}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif target_relative_pos is not None:
            cv2.putText(frame, f"Target Offset: {target_relative_pos:.1f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.tracker.target_track_id is not None:
            # Target ID exists but maybe lost this frame
            cv2.putText(frame, f"Target ID {self.tracker.target_track_id} LOST", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)  # Orange
        else:
            cv2.putText(frame, "No target selected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {self.display_fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display Battery Level
        batt_text = f"Battery: {self.battery_level}%" if self.battery_level is not None else "Battery: N/A"
        batt_color = (0, 255, 0)  # Green default
        if self.battery_level is not None:
            if self.battery_level < 20:
                batt_color = (0, 0, 255)  # Red if low
            elif self.battery_level < 50:
                batt_color = (0, 165, 255)  # Orange if medium
        cv2.putText(frame, batt_text, (10, 90),  # Position below target info
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, batt_color, 2)

        # Display Controls Help
        help_y_start = 30
        help_spacing = 30
        cv2.putText(frame, "Q: Quit", (width - 150, help_y_start),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "SPACE: Takeoff/Land", (width - 220, help_y_start + help_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "F: Toggle Face Mode", (width - 220, help_y_start + 2 * help_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Click: Select Target", (width - 220, help_y_start + 3 * help_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display Drone Status
        status_text = "Status: FLYING" if self.is_flying else "Status: LANDED"
        status_color = (0, 255, 0) if self.is_flying else (0, 0, 255)
        cv2.putText(frame, status_text, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    def run(self):
        if not self.setup():
            print("Setup failed. Exiting.")
            return

        try:
            while True:
                # Update battery periodically
                current_time = time.time()
                if current_time - self.last_battery_check_time > self.battery_check_interval:
                    try:
                        self.battery_level = self.tello.get_battery()
                        self.last_battery_check_time = current_time
                    except:
                        pass  # Ignore battery check errors

                # Get frame
                frame = self.tello.get_frame_read().frame
                if frame is None:
                    print("⚠️ Empty frame received")
                    time.sleep(0.1)
                    continue

                # Convert to RGB for YOLO
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]

                # Tracking logic based on current mode
                confirmed_tracks = []
                target_relative_pos = None

                # Copy the image for drawing
                img_to_draw = img.copy()

                if not self.face_mode:
                    # PERSON TRACKING MODE
                    # 1) YOLO detect + track
                    detections = self.detector.detect(img, conf_threshold=0.30)
                    confirmed_tracks = self.tracker.update(detections)

                    # Get target's relative position for turning
                    target_relative_pos = self.tracker.get_target_relative_horizontal_position(w)

                    # 2) Get target bbox if any
                    box = self.tracker.get_target_info()
                    if box:
                        x1, y1, x2, y2 = map(int, box)

                        # Process ROI for face detection only if we have a target bbox
                        try:
                            # Validate ROI coordinates
                            if y2 > y1 and x2 > x1 and y1 >= 0 and x1 >= 0 and y2 <= h and x2 <= w:
                                roi = img_to_draw[y1:y2, x1:x2]
                                roi_h, roi_w = roi.shape[:2]

                                if roi_h > 0 and roi_w > 0:
                                    # Find face in ROI
                                    face_info = find_face(roi)

                                    if face_info[1] > FACE_LOCK_THRESHOLD:  # Face with sufficient area detected
                                        print(f"🔍 Detected face with area {face_info[1]}, switching to face tracking mode")
                                        self.face_mode = True
                                        self.face_roi = [x1, y1, x2, y2]

                                        # Store face center in global coordinates
                                        (face_x, face_y), face_area = face_info
                                        abs_face_x = face_x + x1
                                        abs_face_y = face_y + y1

                                        # Draw face circle
                                        cv2.circle(img_to_draw, (abs_face_x, abs_face_y), 8, (255, 0, 255), -1)

                                        # Initialize tracking variables
                                        self.last_face_info = ([abs_face_x, abs_face_y], face_area)
                                        self.prev_error = (abs_face_x - w//2, abs_face_y - h//2)
                                        self.face_lost_counter = 0
                                        continue  # Skip to next frame to start face tracking

                                    elif face_info[1] > 0:  # Face detected but not large enough for lock
                                        # Draw face with current coordinates
                                        (cx, cy), area = face_info
                                        abs_cx, abs_cy = cx + x1, cy + y1
                                        cv2.circle(img_to_draw, (abs_cx, abs_cy), 5, (255, 0, 0), -1)

                                        # Use basic person tracking (yaw only)
                                        center_x = (x1 + x2) // 2 - w // 2  # Relative to center
                                        yaw = int(np.clip(0.1 * center_x, -30, 30))
                                        self.tello.send_rc_control(0, 0, 0, yaw)
                                    else:
                                        # No face in person bounding box, use center of bbox for basic tracking
                                        center_x = (x1 + x2) // 2 - w // 2  # Relative to center
                                        yaw = int(np.clip(0.1 * center_x, -30, 30))
                                        self.tello.send_rc_control(0, 0, 0, yaw)
                            else:
                                # Invalid ROI coordinates
                                self.tello.send_rc_control(0, 0, 0, 0)
                        except Exception as e:
                            print(f"❌ Error processing person ROI: {e}")
                            self.tello.send_rc_control(0, 0, 0, 0)
                    else:
                        # No target selected - hover
                        self.tello.send_rc_control(0, 0, 0, 0)

                else:
                    # FACE-ONLY TRACKING MODE
                    try:
                        # Use the stored ROI or expand it slightly for tracking robustness
                        if self.face_roi is None:
                            # No face ROI - use full frame
                            roi = img_to_draw
                            roi_x1, roi_y1 = 0, 0
                        else:
                            # Get stored ROI with optional expansion for tracking robustness
                            x1, y1, x2, y2 = self.face_roi
                            x1, y1, x2, y2 = expand_roi(x1, y1, x2, y2, w, h, 1.3)

                            # Extract ROI and store offset
                            roi = img_to_draw[y1:y2, x1:x2]
                            roi_x1, roi_y1 = x1, y1

                        # Detect face in ROI
                        face_info = find_face(roi)

                        if face_info[1] > 0:  # Face detected
                            # Reset counter since we found a face
                            self.face_lost_counter = 0

                            # Get face coordinates and map to full frame
                            (face_x, face_y), face_area = face_info
                            abs_face_x = face_x + roi_x1
                            abs_face_y = face_y + roi_y1

                            # Update face ROI around new position for next frame
                            face_size = int(np.sqrt(face_area))
                            self.face_roi = [
                                max(0, abs_face_x - face_size),
                                max(0, abs_face_y - face_size),
                                min(w, abs_face_x + face_size),
                                min(h, abs_face_y + face_size)
                            ]

                            # Store for drawing
                            self.last_face_info = ([abs_face_x, abs_face_y], face_area)

                            # Draw face circle
                            cv2.circle(img_to_draw, (abs_face_x, abs_face_y), 8, (255, 0, 255), -1)

                            # Track face directly using PID controller
                            # Calculate center-relative error
                            x_err = abs_face_x - w//2
                            y_err = abs_face_y - h//2

                            # Run PID tracking with face coordinates
                            self.prev_error = track_face(
                                self.tello,
                                ([x_err, y_err], face_area),  # using error from center
                                self.pid,
                                self.prev_error,
                                self.fbRange
                            )

                        else:
                            # Face lost - increment counter
                            self.face_lost_counter += 1
                            print(f"⚠️ Face lost for {self.face_lost_counter} frames")

                            if self.face_lost_counter >= FACE_LOST_COUNT:
                                # Switch back to person tracking if face lost for too long
                                print("🔍 Face lost for too long, reverting to person tracking mode")
                                self.face_mode = False
                                self.face_roi = None
                                self.face_lost_counter = 0

                                # Stop movement temporarily during mode switch
                                self.tello.send_rc_control(0, 0, 0, 0)
                            else:
                                # Temporarily lost face but staying in face mode - keep last ROI and hover
                                self.tello.send_rc_control(0, 0, 0, 0)

                    except Exception as e:
                        print(f"❌ Error in face tracking mode: {e}")
                        self.face_lost_counter += 1

                        if self.face_lost_counter >= FACE_LOST_COUNT:
                            # Revert to person tracking on error
                            print("🔍 Face tracking error, reverting to person tracking mode")
                            self.face_mode = False
                            self.face_roi = None

                        # Hover safely on error
                        self.tello.send_rc_control(0, 0, 0, 0)

                # Display frame with overlay
                disp = cv2.cvtColor(img_to_draw, cv2.COLOR_RGB2BGR)

                # Draw all the overlay elements
                self._draw_overlay(disp, confirmed_tracks, target_relative_pos)

                # Calculate FPS
                self.frame_count_fps += 1
                if current_time - self.last_fps_time >= 1.0:
                    self.display_fps = self.frame_count_fps / (current_time - self.last_fps_time)
                    self.frame_count_fps = 0
                    self.last_fps_time = current_time

                # Show image
                cv2.imshow(self.window_name, disp)

                # Process key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("'Q' pressed. Exiting...")
                    break
                elif key == ord('f'):
                    # Toggle face tracking mode manually
                    self.face_mode = not self.face_mode
                    print(f"Manually toggled face mode: {'ON' if self.face_mode else 'OFF'}")
                    if not self.face_mode:
                        self.face_roi = None
                elif key == ord(' '):  # Spacebar
                    if not self.is_flying:
                        print("Taking off...")
                        try:
                            self.tello.takeoff()
                            self.is_flying = True
                        except Exception as e:
                            print(f"❌ Takeoff failed: {e}")
                    else:
                        print("Landing...")
                        try:
                            # Stop movement before landing
                            self.tello.send_rc_control(0, 0, 0, 0)
                            time.sleep(0.5)
                            self.tello.land()
                            self.is_flying = False
                        except Exception as e:
                            print(f"❌ Landing failed: {e}")

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Landing...")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            print("Cleaning up...")
            try:
                # Make sure drone stops moving
                self.tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)

                # Land if flying
                if self.is_flying:
                    try:
                        # Try landing multiple times
                        for attempt in range(3):
                            try:
                                self.tello.land()
                                print("Landing successful")
                                break
                            except Exception as e:
                                print(f"Landing attempt {attempt+1} failed: {e}")
                                time.sleep(1)
                    except:
                        print("Failed to land after multiple attempts")

                # Stop stream and end connection
                try:
                    if hasattr(self.tello, 'is_stream_on') and self.tello.is_stream_on:
                        self.tello.streamoff()
                except:
                    pass

                try:
                    self.tello.end()
                except:
                    pass

            except Exception as e:
                print(f"❌ Error during cleanup: {e}")

            cv2.destroyAllWindows()
            print("Cleanup complete")

if __name__ == "__main__":
    DroneFaceTracker().run()
