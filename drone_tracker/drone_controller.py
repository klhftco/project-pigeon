import time
import cv2
from djitellopy import Tello
import numpy as np
import logging # Added for logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the new classes
from .yolo_detector import YoloDetector
from .tracker import SimpleTracker

class DroneController:
    """Manages drone connection, control, and integrates tracking."""

    def __init__(self,
                 yolo_model_path="yolov8n.pt",
                 conf_threshold=0.30,
                 iou_threshold=0.3,
                 max_track_age=5,
                 min_track_hits=3,
                 horizontal_threshold=15.0, # Target % offset threshold for turning
                 max_yaw_speed=20,
                 kp_yaw=0.8,             # Proportional control gain for yaw
                 battery_check_interval=5, # Seconds between battery checks
                 command_interval=0.2,
                 tello=None): # Minimum seconds between sending commands
        """
        Initializes the drone controller, detector, and tracker.
        """
        logging.info("Initializing DroneController...")
        self.tello = tello
        if not tello:
            self.tello = Tello()
        self.detector = YoloDetector(model_path=yolo_model_path)
        self.tracker = SimpleTracker(iou_threshold=iou_threshold,
                                     max_age=max_track_age,
                                     min_hits=min_track_hits)

        self.conf_threshold = conf_threshold
        self.horizontal_threshold = horizontal_threshold
        self.max_yaw_speed = max_yaw_speed
        self.kp_yaw = kp_yaw

        self.is_flying = False
        self.frame_reader = None
        self.window_name = "Drone Tracker View"
        self.last_fps_time = time.time()
        self.frame_count_fps = 0
        self.display_fps = 0.0
        self._setup_complete = False

        # Battery tracking
        self.battery_level = None
        self.last_battery_check_time = 0
        self.battery_check_interval = battery_check_interval # Use the parameter

        # Command rate limiting
        self.command_interval = command_interval
        self.last_command_time = 0 # Initialize last command time
        self.frame_skip_counter = 0 # Initialize frame skip counter

    def _connect_and_setup(self):
        """Connects to the drone, starts stream, sets up CV window."""
        try:
            logging.info("Connecting to Tello drone...")
            self.tello.connect()
            # Get initial battery level
            self.battery_level = self.tello.get_battery()
            self.last_battery_check_time = time.time() # Record time of initial check
            logging.info(f"✅ Drone connected. Battery: {self.battery_level}%")

            logging.info("Starting video stream...")
            self.tello.streamon()
            self.frame_reader = self.tello.get_frame_read()
            # Allow time for stream to stabilize
            time.sleep(2)
            logging.info("✅ Video stream started.")

            # Create window and set mouse callback *after* stream starts
            # Ensures frame dimensions are available if needed by callback initially
            cv2.namedWindow(self.window_name)
            # Pass the tracker's method as the callback
            cv2.setMouseCallback(self.window_name, self.tracker.handle_mouse_click)
            logging.info("✅ OpenCV window created and mouse callback set.")
            self._setup_complete = True
            return True

        except Exception as e:
            logging.error(f"❌ Error during drone connection or setup: {e}", exc_info=True)
            logging.warning("   Check drone connection, power, and network.")
            # Clean up if stream started but window failed etc.
            try:
                self.tello.streamoff()
            except Exception:
                pass # Ignore errors during cleanup
            cv2.destroyAllWindows()
            return False

    def connect_to_drone(self):
        """Connects to the Tello drone and retrieves battery info."""
        try:
            logging.info("Connecting to Tello drone...")
            self.tello.connect()
            self.battery_level = self.tello.get_battery()
            self.last_battery_check_time = time.time()
            logging.info(f"✅ Drone connected. Battery: {self.battery_level}%")
            return True
        except Exception as e:
            logging.error(f"❌ Error connecting to drone: {e}", exc_info=True)
            return False

    def setup_video_and_ui(self):
        """Starts video stream and sets up OpenCV window and UI callbacks."""
        try:
            logging.info("Starting video stream...")
            self.tello.streamon()
            self.frame_reader = self.tello.get_frame_read()
            time.sleep(2)
            logging.info("✅ Video stream started.")

            # Create window and set mouse callback
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self.tracker.handle_mouse_click)
            logging.info("✅ OpenCV window created and mouse callback set.")
            self._setup_complete = True
            return True
        except Exception as e:
            logging.error(f"❌ Error during video/UI setup: {e}", exc_info=True)
            logging.warning("Check if video stream is already running or UI is blocked.")
            try:
                self.tello.streamoff()
            except Exception:
                pass
            cv2.destroyAllWindows()
            return False

    def _calculate_yaw_speed(self, target_relative_pos_percent):
        """
        Calculates the yaw speed based on the target's relative horizontal position.
        Uses proportional control based on the error outside the threshold.
        """
        if target_relative_pos_percent is None:
            return 0

        yaw_speed = 0
        # Only turn if target is outside threshold
        if abs(target_relative_pos_percent) > self.horizontal_threshold:
            # Calculate error (how far outside the threshold the target is)
            # Ensure error is positive for calculation
            if target_relative_pos_percent > 0: # Target is to the right
                error = target_relative_pos_percent - self.horizontal_threshold
            else: # Target is to the left
                error = abs(target_relative_pos_percent) - self.horizontal_threshold

            # Calculate yaw speed using proportional control
            yaw_speed = int(self.kp_yaw * error)

            # Clip to maximum speed
            yaw_speed = min(yaw_speed, self.max_yaw_speed)

            # Set direction (negative = left, positive = right)
            if target_relative_pos_percent < 0:
                yaw_speed = -yaw_speed
                # print(f"Turning LEFT with speed {abs(yaw_speed)}") # Optional debug print
            # else:
                # print(f"Turning RIGHT with speed {yaw_speed}") # Optional debug print
        # else:
            # print("Target centered - no turning needed") # Optional debug print

        return yaw_speed

    def _draw_overlay(self, frame, confirmed_tracks, target_relative_pos):
        """
        Draws tracking boxes, target info, FPS, controls help, and battery onto the frame.
        """
        height, width = frame.shape[:2]

        # Draw all confirmed tracks
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
                color = (0, 255, 255) # Bright Yellow
                thickness = 3
                label += " [SELECTED]"
            elif is_target:
                color = (255, 255, 0) # Bright Cyan
                thickness = 3
                label += " [TARGET]"

            cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
            cv2.putText(frame, label, (x0, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

        # Draw Center Line and Threshold Lines
        center_x = width // 2
        threshold_px = int(center_x * (self.horizontal_threshold / 100.0))
        cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 255), 1) # Center line (Red)
        cv2.line(frame, (center_x - threshold_px, 0),
                 (center_x - threshold_px, height), (255, 0, 0), 1) # Left Threshold (Blue)
        cv2.line(frame, (center_x + threshold_px, 0),
                 (center_x + threshold_px, height), (255, 0, 0), 1) # Right Threshold (Blue)

        # Display Target Info
        if target_relative_pos is not None:
             cv2.putText(frame, f"Target Offset: {target_relative_pos:.1f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif self.tracker.target_track_id is not None:
            # Target ID exists but maybe lost this frame
            cv2.putText(frame, f"Target ID {self.tracker.target_track_id} LOST", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Orange
        else:
             cv2.putText(frame, "No target selected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {self.display_fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display Battery Level
        batt_text = f"Battery: {self.battery_level}%" if self.battery_level is not None else "Battery: N/A"
        batt_color = (0, 255, 0) # Green default
        if self.battery_level is not None:
            if self.battery_level < 20:
                batt_color = (0, 0, 255) # Red if low
            elif self.battery_level < 50:
                batt_color = (0, 165, 255) # Orange if medium
        cv2.putText(frame, batt_text, (10, 90), # Position below target info
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, batt_color, 2)

        # Display Controls Help
        help_y_start = 30
        help_spacing = 30
        cv2.putText(frame, "Q: Quit", (width - 150, help_y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "SPACE: Takeoff/Land", (width - 220, help_y_start + help_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Click: Select Target", (width - 220, help_y_start + 2 * help_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display Drone Status
        status_text = "Status: FLYING" if self.is_flying else "Status: LANDED"
        status_color = (0, 255, 0) if self.is_flying else (0, 0, 255)
        cv2.putText(frame, status_text, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)


    def run(self, connected=False):
        """Runs the main control loop."""
        if not connected:
            if not self._connect_and_setup():
                logging.error("❌ Failed to initialize drone connection. Exiting.")
                return # Exit if setup failed
        else:
            self.setup_video_and_ui()

        update_interval = 0.05 # Target 20 FPS processing loop (adjust as needed)

        logging.info("Starting main control loop. Press SPACE to takeoff/land, Q to quit.")

        try:
            while True:
                current_time = time.time() # Get current time once per loop

                # --- Periodic Battery Check ---
                if current_time - self.last_battery_check_time > self.battery_check_interval:
                    try:
                        # Update Battery
                        new_battery_level = self.tello.get_battery()
                        if new_battery_level != self.battery_level:
                             # logging.debug(f"Battery updated: {new_battery_level}%") # Optional debug
                             self.battery_level = new_battery_level

                        # Update Flight Status based on drone report
                        actual_is_flying = self.tello.is_flying
                        if actual_is_flying != self.is_flying:
                             logging.info(f"Drone reported flight status change: {'Flying' if actual_is_flying else 'Landed'}. Updating state.")
                             self.is_flying = actual_is_flying # Correct the state based on drone report

                        self.last_battery_check_time = current_time

                    except Exception as e:
                        logging.warning(f"⚠️ Could not update battery level or flight status: {e}")
                        # Keep the old values, maybe it will recover

                loop_start_time = current_time # Use the time already fetched

                # 1. Get Frame
                frame = self.frame_reader.frame
                if frame is None:
                    logging.warning("⚠️ Warning: Received empty frame. Skipping iteration.")
                    # Need to handle key input and sleep even on empty frame
                    key = cv2.waitKey(1) & 0xFF # Check keys even if frame is bad
                    if key == ord('q') or key == 27:
                        logging.info("'Q' pressed (empty frame). Landing and exiting...")
                        break
                    # Add spacebar handling here if needed, similar to below

                    time.sleep(0.1) # Avoid busy-waiting on empty frames
                    elapsed_time = time.time() - loop_start_time
                    sleep_time = update_interval - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                # --- Frame Processing (Conditional) ---
                self.frame_skip_counter += 1
                frame_processed = False # Flag to track if frame was processed this iteration

                if self.frame_skip_counter % 2 != 0: # Process odd frames (1, 3, 5...)
                    frame_processed = True
                    # 1b. Prepare Frame
                    frame = frame[:, :, ::-1] # Swap BGR/RGB
                    height, width = frame.shape[:2]
                    img_to_draw_on = frame.copy()

                    # 2. Detect Objects
                    detections = self.detector.detect(frame, conf_threshold=self.conf_threshold)

                    # 3. Update Tracker
                    confirmed_tracks = self.tracker.update(detections)

                    # 4. Get Target Info
                    target_relative_pos = self.tracker.get_target_relative_horizontal_position(width)

                    # 5. Calculate & Send Control Command (rate-limited)
                    if current_time - self.last_command_time >= self.command_interval:
                        yaw_speed = 0
                        if self.is_flying:
                            yaw_speed = self._calculate_yaw_speed(target_relative_pos)
                            # Send command only if flying
                            self.tello.send_rc_control(0, 0, 0, yaw_speed)
                            self.last_command_time = current_time # Update last command time
                        else:
                            # Ensure drone doesn't drift if landed
                            self.tello.send_rc_control(0, 0, 0, 0)
                            self.last_command_time = current_time # Update last command time

                    # 6. Draw Overlay
                    self._draw_overlay(img_to_draw_on, confirmed_tracks, target_relative_pos)

                    # 7. Display Frame
                    cv2.imshow(self.window_name, img_to_draw_on)

                    # 9a. Update Processed Frame Counter
                    self.frame_count_fps += 1

                # --- Operations on Every Loop Iteration (Post-Processing) ---

                # 8. Handle Keyboard Input (Always)
                # Use waitKey(1) even if frame wasn't displayed this iteration to catch key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # q or ESC
                    logging.info("'Q' pressed. Landing and exiting...")
                    break
                elif key == ord(' '): # Spacebar
                    log_prefix = "(Processed Frame)" if frame_processed else "(Skipped Frame)"
                    if not self.is_flying:
                        logging.info(f"Spacebar pressed {log_prefix}. Attempting takeoff...")
                        try:
                            self.tello.takeoff()
                            self.is_flying = True
                            logging.info("✅ Takeoff successful.")
                        except Exception as e:
                            logging.error(f"❌ Takeoff failed: {e}")
                    else:
                        logging.info(f"Spacebar pressed {log_prefix}. Attempting landing...")
                        try:
                            # Ensure drone is hovering before landing
                            self.tello.send_rc_control(0, 0, 0, 0)
                            time.sleep(0.5) # Give it a moment to stabilize
                            self.tello.land()
                            self.is_flying = False
                            logging.info("✅ Landing successful.")
                        except Exception as e:
                            logging.warning(f"❌ Landing failed: {e}")
                            # Keep is_flying as True if land command failed

                # 9b. Update FPS Display Calculation (Always, based on processed frames)
                if current_time - self.last_fps_time >= 1.0: # Update FPS display every second
                    self.display_fps = self.frame_count_fps / (current_time - self.last_fps_time)
                    self.frame_count_fps = 0 # Reset counter for the next second
                    self.last_fps_time = current_time

                # 10. Calculate Sleep Time (Always)
                elapsed_time = time.time() - loop_start_time
                sleep_time = update_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # else: # Optional: log if loop is taking too long
                #     logging.debug(f"Loop time ({elapsed_time:.3f}s) exceeded target interval ({update_interval:.3f}s).")

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt detected. Landing and exiting...")
        except Exception as e:
             logging.error(f"\n❌ An unexpected error occurred in the main loop: {e}", exc_info=True)
        finally:
            logging.info("Cleaning up...")
            # Ensure drone lands if it was flying
            if self.is_flying:
                logging.warning("Drone was flying. Attempting emergency land.")
                try:
                    self.tello.send_rc_control(0,0,0,0) # Stop movement
                    self.tello.land()
                except Exception as land_err:
                    logging.error(f"Emergency land failed: {land_err}")

            # Turn off stream and RC control regardless of flight status
            logging.info("Turning off RC control and video stream...")
            try:
                 # Send zero movement command one last time
                 self.tello.send_rc_control(0,0,0,0)
                 time.sleep(0.1)
                 if self.tello.is_stream_on:
                      self.tello.streamoff()
            except Exception as clean_err:
                 logging.warning(f"⚠️ Error during stream/RC off: {clean_err}")

            # Explicitly end connection
            try:
                 self.tello.end()
                 logging.info("Tello connection closed.")
            except Exception as end_err:
                 logging.warning(f"⚠️ Error closing Tello connection: {end_err}")

            cv2.destroyAllWindows()
            logging.info("✅ Cleanup finished. Exiting.")

# Main execution block
if __name__ == '__main__':
    # To run this directly, ensure you are in the parent directory
    # (e.g., local_owl) and run using: python -m drone_tracker.drone_controller
    controller = DroneController(
        yolo_model_path="yolov8n.pt", # Make sure this path is correct
        conf_threshold=0.35,
        iou_threshold=0.3,
        max_track_age=10, # Increased persistence
        min_track_hits=3,
        horizontal_threshold=10.0, # Tighter threshold
        max_yaw_speed=25, # Slightly faster turn
        kp_yaw=0.7,       # Adjusted gain
        battery_check_interval=5, # Check battery every 5 seconds
        command_interval=0.2 # Send commands max every 200ms
    )
    controller.run()
