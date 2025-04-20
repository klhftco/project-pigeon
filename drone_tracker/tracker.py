import numpy as np
import cv2 # Needed for mouse callback interaction
import math # Added for distance calculation

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    # box format: [x1, y1, x2, y2]
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def calculate_center(box):
    """Calculates the center (x, y) of a bounding box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def calculate_area(box):
    """Calculates the area of a bounding box."""
    return (box[2] - box[0]) * (box[3] - box[1])

class SimpleTracker:
    """Manages tracking detected objects using simple IoU matching."""

    def __init__(self, iou_threshold=0.4, max_age=10, min_hits=3, reacquisition_max_distance=75, reacquisition_size_tolerance=0.5):
        """
        Initializes the tracker.

        Args:
            iou_threshold (float): Minimum IoU to match a detection to a track.
            max_age (int): Max frames a track can exist without a detection before being removed or re-acquired.
            min_hits (int): Min detections needed to confirm a track.
            reacquisition_max_distance (float): Max pixel distance between lost target center and candidate center for re-acquisition.
            reacquisition_size_tolerance (float): Allowed fractional difference in area for re-acquisition (e.g., 0.5 means area can be 50% smaller or larger).
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.reacquisition_max_distance = reacquisition_max_distance
        self.reacquisition_size_tolerance = reacquisition_size_tolerance
        self.active_tracks = []
        self.next_track_id = 0
        self.selected_track_id = None # Manually selected by user click
        self.target_track_id = None   # Automatically targeted (or can be set manually)
        self.frame_count = 0          # Internal frame counter
        self.last_known_target_info = None # Store info about the target just before it was lost

    def update(self, current_detections):
        """
        Updates the tracker state with new detections.

        Args:
            current_detections (list): List of detections from the current frame.
                                       Each detection is [[x1, y1, x2, y2], score].

        Returns:
            list: List of currently active and confirmed tracks.
                  Each track is a dict: {'id', 'box', 'age', 'hits'}.
        """
        self.frame_count += 1

        # --- Matching Logic ---
        matched_indices = set()
        track_indices = list(range(len(self.active_tracks)))
        detection_indices = list(range(len(current_detections)))
        unmatched_detections = [] # Keep track of detections not matched by IoU

        if self.active_tracks and current_detections:
            iou_matrix = np.zeros((len(self.active_tracks), len(current_detections)))
            for t_idx, track in enumerate(self.active_tracks):
                for d_idx, detection in enumerate(current_detections):
                    iou_matrix[t_idx, d_idx] = calculate_iou(track['box'], detection[0])

            # Simple greedy matching
            for _ in range(min(len(self.active_tracks), len(current_detections))):
                if iou_matrix.size == 0 or np.max(iou_matrix) < self.iou_threshold:
                    break

                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)

                track = self.active_tracks[t_idx]
                detection_box = current_detections[d_idx][0]
                track['box'] = detection_box
                track['age'] = 0
                track['hits'] += 1
                
                # If this matched track was the target, clear any last known info
                if track['id'] == self.target_track_id:
                    self.last_known_target_info = None

                matched_indices.add(d_idx)
                if t_idx in track_indices: # Ensure index exists before removing
                    track_indices.remove(t_idx)

                # Prevent rematching this track or detection
                iou_matrix[t_idx, :] = -1
                iou_matrix[:, d_idx] = -1

        # Store unmatched detections for potential re-acquisition
        all_detection_indices = set(range(len(current_detections)))
        unmatched_detection_indices = all_detection_indices - matched_indices
        unmatched_detections = [(idx, current_detections[idx]) for idx in unmatched_detection_indices]

        # --- Handle Unmatched Tracks ---
        lost_target_reacquired = False
        lost_target_info = None # Store info if target is lost

        # First pass: Keep matched tracks and increment age for unmatched ones
        temp_active_tracks = []
        indices_of_potentially_lost_tracks = []
        for t_idx in range(len(self.active_tracks)):
            track = self.active_tracks[t_idx]
            is_matched = t_idx not in track_indices # Check if this track index was matched earlier
            if is_matched:
                 temp_active_tracks.append(track) # Keep matched track
            else:
                # This track was not matched by IoU
                track['age'] += 1
                if track['age'] <= self.max_age:
                    temp_active_tracks.append(track) # Keep aged track if not too old
                else:
                    # Track is older than max_age - potentially lost
                    # indices_of_potentially_lost_tracks.append(t_idx) # Not strictly needed anymore
                    if track['id'] == self.target_track_id:
                        lost_target_info = track.copy() # Save lost target info for re-acquisition attempt
                        # Store the info *before* attempting re-acquisition, might be needed for auto-select later
                        self.last_known_target_info = lost_target_info
                        print(f"ℹ️ Tracker: Target track ID {self.target_track_id} exceeded max_age. Attempting re-acquisition...")
                    elif track['id'] == self.selected_track_id:
                        print(f"ℹ️ Tracker: Lost selected track ID: {self.selected_track_id} (aged out)")
                        self.selected_track_id = None # Deselect if aged out

        # --- Attempt Target Re-acquisition ---
        reacquired_detection_idx = -1
        if lost_target_info and unmatched_detections:
            best_candidate = None
            min_score = float('inf')
            lost_center = calculate_center(lost_target_info['box'])
            lost_area = calculate_area(lost_target_info['box'])

            for det_idx, detection in unmatched_detections:
                det_box = detection[0]
                det_center = calculate_center(det_box)
                det_area = calculate_area(det_box)

                # Calculate distance
                distance = math.dist(lost_center, det_center)

                # Calculate size similarity (avoid division by zero)
                if lost_area > 1e-6 and det_area > 1e-6:
                    size_ratio = det_area / lost_area
                    size_diff = abs(1 - size_ratio)
                else:
                    size_diff = float('inf') # Penalize zero area boxes

                # Check thresholds
                if distance < self.reacquisition_max_distance and size_diff < self.reacquisition_size_tolerance:
                    # Simple scoring: prioritize distance
                    score = distance # Lower score is better
                    if score < min_score:
                        min_score = score
                        best_candidate = (det_idx, detection)

            if best_candidate:
                reacquired_detection_idx, reacquired_detection = best_candidate
                print(f"✅ Tracker: Re-acquired target ID {self.target_track_id} with detection index {reacquired_detection_idx}.")
                new_target_track = {
                    'id': self.target_track_id,
                    'box': reacquired_detection[0],
                    'age': 0,
                    'hits': lost_target_info.get('hits', self.min_hits) # Reset age, keep hits (or min_hits)
                }
                temp_active_tracks.append(new_target_track) # Add the re-acquired track
                lost_target_reacquired = True
                self.last_known_target_info = None # Clear last known info on successful re-acquisition
                # Remove the re-acquired detection from the list so it doesn't become a new track
                unmatched_detections = [(idx, det) for idx, det in unmatched_detections if idx != reacquired_detection_idx]

        # Finalize active tracks list
        self.active_tracks = temp_active_tracks

        # If target was lost and *not* re-acquired, clear the target ID
        if lost_target_info and not lost_target_reacquired:
            lost_id = lost_target_info['id'] # Store ID before potentially clearing
            print(f"❌ Tracker: Failed to re-acquire target ID {lost_id}. No suitable candidate found.")
            # Keep self.last_known_target_info - it was set before re-acquisition attempt
            self.target_track_id = None # Clear the target ID

            # If the lost target was the user-selected one, clear selection
            if self.selected_track_id == lost_id:
                print(f"ℹ️ Tracker: Lost user-selected target ID {lost_id}.")
                self.selected_track_id = None

            # Auto-selection will now happen at the end using last_known_target_info if available

        # --- Handle Remaining Unmatched Detections (New Tracks) ---
        for d_idx, detection in unmatched_detections:
             # Check if this detection index was already used for re-acquisition (double check)
             if d_idx == reacquired_detection_idx:
                 continue

             new_track = {
                 'id': self.next_track_id,
                 'box': detection[0],
                 'age': 0,
                 'hits': 1
             }
             self.active_tracks.append(new_track)
             self.next_track_id += 1
             # If this new track becomes the only track and no target exists,
             # auto-select might pick it up immediately below.

        # --- Final Target Check & Auto-Selection ---
        # If, after all updates, there is still no target, try to auto-select one.
        if self.target_track_id is None:
            # Pass the last known info (if any) to guide selection
            self.auto_select_target()

        # Return only confirmed tracks
        return [t for t in self.active_tracks if t['hits'] >= self.min_hits]

    def auto_select_target(self):
        """Selects a target automatically.
        If last_known_target_info is available, selects the confirmed track closest
        to the last known position. Otherwise, selects the largest confirmed track.
        Should be called when self.target_track_id is None.
        """
        confirmed_tracks = [t for t in self.active_tracks if t['hits'] >= self.min_hits]
        if not confirmed_tracks:
            # Optional: print statement if needed
            # print("ℹ️ Tracker: Auto-select failed. No confirmed tracks available.")
            # Ensure last known info is cleared if we fail to select anything
            self.last_known_target_info = None
            return # Nothing to select

        selected_track = None
        if self.last_known_target_info:
            print(f"ℹ️ Tracker: Attempting auto-selection based on proximity to lost target ID {self.last_known_target_info['id']}...")
            lost_center = calculate_center(self.last_known_target_info['box'])
            # lost_area = calculate_area(self.last_known_target_info['box']) # Keep for potential future size weighting

            min_dist = float('inf')
            best_candidate = None

            for track in confirmed_tracks:
                candidate_center = calculate_center(track['box'])
                distance = math.dist(lost_center, candidate_center)
                # candidate_area = calculate_area(track['box'])
                # size_diff = abs(lost_area - candidate_area) # Keep for potential future weighting

                # Simple proximity selection
                if distance < min_dist:
                    min_dist = distance
                    best_candidate = track

            if best_candidate:
                # Optionally add a max distance threshold? For now, just closest.
                selected_track = best_candidate
                print(f"🎯 Tracker: Auto-selected closest confirmed track to lost target: ID {selected_track['id']} (Distance: {min_dist:.2f})")
            else:
                print(f"ℹ️ Tracker: Could not find a suitable close track based on last known position.")
                # Fall through to largest track selection below

        # If no selection based on proximity (no last_known_info or no suitable candidate found), fall back to largest
        if selected_track is None:
            print(f"ℹ️ Tracker: Falling back to selecting largest confirmed track...")
            largest_track = max(confirmed_tracks, key=lambda t: calculate_area(t['box']))
            selected_track = largest_track
            print(f"🎯 Tracker: Auto-selected largest track as target: ID {selected_track['id']}")

        # Final assignment
        self.target_track_id = selected_track['id']
        # Auto-selection clears manual selection focus, but keeps target
        self.selected_track_id = None
        # Clear the info used for this selection attempt
        self.last_known_target_info = None

    def handle_mouse_click(self, event, x, y, flags, param):
        """
        Callback function for mouse events to select tracks.
        To be registered with cv2.setMouseCallback.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_track_id = None
            min_area = float('inf')

            confirmed_tracks = [t for t in self.active_tracks if t['hits'] >= self.min_hits]

            for track in confirmed_tracks:
                x0, y0, x1, y1 = map(int, track['box'])
                if x0 <= x <= x1 and y0 <= y <= y1:
                    area = (x1 - x0) * (y1 - y0)
                    if area < min_area:
                         min_area = area
                         clicked_track_id = track['id']

            if clicked_track_id is not None:
                self.selected_track_id = clicked_track_id
                self.target_track_id = clicked_track_id # Also set as target
                self.last_known_target_info = None # Clear last known info on manual selection
                print(f"🖱️ Tracker: Click selected Track ID: {self.selected_track_id} as new target.")
            else:
                 # Clicked outside any confirmed track, deselect
                 print(f"🖱️ Tracker: Clicked outside any track. Deselecting.")
                 self.selected_track_id = None
                 self.target_track_id = None # Also clear target if clicking background
                 self.last_known_target_info = None # Clear last known info if deselecting target

    def get_target_info(self):
        """
        Returns the box of the currently targeted track.

        Returns:
            list or None: The bounding box [x1, y1, x2, y2] of the target track,
                          or None if no target is selected or found.
        """
        if self.target_track_id is None:
            return None

        for track in self.active_tracks:
            if track['id'] == self.target_track_id:
                 # Check age again just in case, although re-acquisition handles the primary loss
                 if track['age'] <= self.max_age:
                     return track['box']
                 else:
                     # This case should be less common now due to re-acquisition logic,
                     # but acts as a fallback if target is lost and immediately queried.
                     # Re-acquisition logic already handles setting target_track_id to None if needed.
                     # We might reach here if get_target_info is called *before* the next update() cycle
                     # where re-acquisition would fail.
                     print(f"ℹ️ Tracker: Target track {track['id']} is currently aged out (age {track['age']} > {self.max_age}). Awaiting next update cycle for potential re-acquisition or final loss.")
                     # It's better to return None than an old box if it's currently past max_age.
                     # The target_track_id might still be set, waiting for re-acquisition attempt.
                     return None

        # Target ID is set, but track not found in active list (could happen if removed abruptly)
        print(f"⚠️ Tracker: Target track ID {self.target_track_id} not found in active tracks list during get_target_info(). Clearing target.")
        self.target_track_id = None
        # Clear selected if it was the same as the lost target
        if self.selected_track_id is not None and self.selected_track_id not in [t['id'] for t in self.active_tracks]:
             self.selected_track_id = None
        self.last_known_target_info = None # Clear info if target disappears unexpectedly
        return None

    def get_target_relative_horizontal_position(self, frame_width):
        """
        Calculates the horizontal center of the target bounding box relative 
        to the frame center, expressed as a percentage (-100% to +100%).

        Args:
            frame_width (int): The width of the video frame.

        Returns:
            float or None: The relative horizontal position percentage, or None if
                           no target is available or frame_width is invalid.
        """
        target_box = self.get_target_info()
        if target_box is None or frame_width <= 0:
            return None

        # Calculate center of the target box
        x1, _, x2, _ = target_box
        target_center_x = (x1 + x2) / 2

        # Calculate frame center
        frame_center_x = frame_width / 2

        # Calculate relative position percentage
        # Offset from center / half-width * 100
        # A positive value means the target is to the right of the center.
        # A negative value means the target is to the left of the center.
        relative_position = ((target_center_x - frame_center_x) / frame_center_x) * 100 # Changed from center_x in original controller to frame_center_x

        return relative_position

# Example Usage (if you want to test this file directly)
if __name__ == '__main__':
    print("Testing SimpleTracker...")
    tracker = SimpleTracker()

    # Simulate detections (replace with actual detector output)
    # Format: [[[x1, y1, x2, y2], score], ...]
    detections_frame1 = [
        [[50, 50, 150, 150], 0.9],
        [[200, 200, 300, 300], 0.85]
    ]
    detections_frame2 = [
        [[60, 60, 160, 160], 0.92], # Track 0 should match
        [[350, 350, 450, 450], 0.8] # New track
    ]
    detections_frame3 = [
        [[70, 70, 170, 170], 0.91], # Track 0
        [[360, 360, 460, 460], 0.82] # Track 2
    ]
    detections_frame4 = [
        [[370, 370, 470, 470], 0.85] # Track 2, Track 0 lost
    ]

    print("--- Frame 1 ---")
    confirmed_tracks = tracker.update(detections_frame1)
    print(f"Active Tracks: {tracker.active_tracks}")
    print(f"Confirmed Tracks: {confirmed_tracks}")

    print("\n--- Frame 2 ---")
    confirmed_tracks = tracker.update(detections_frame2)
    print(f"Active Tracks: {tracker.active_tracks}")
    print(f"Confirmed Tracks: {confirmed_tracks}")

    print("\n--- Frame 3 ---")
    confirmed_tracks = tracker.update(detections_frame3)
    print(f"Active Tracks: {tracker.active_tracks}")
    print(f"Confirmed Tracks: {confirmed_tracks}") # Track 0 & 2 should be confirmed now

    print("\n--- Frame 4 (Track 0 lost) ---")
    confirmed_tracks = tracker.update(detections_frame4)
    print(f"Active Tracks: {tracker.active_tracks}")
    print(f"Confirmed Tracks: {confirmed_tracks}")

    print("\n--- Simulating Auto Target Selection (force frame count) ---")
    tracker.frame_count = 9 # Set frame count to trigger auto-target on next update
    detections_frame10 = [
        [[80, 80, 180, 180], 0.9],  # Assume this is track 0 reappearing
        [[380, 380, 480, 480], 0.8], # Track 2
    ]
    print("--- Frame 10 ---")
    confirmed_tracks = tracker.update(detections_frame10)
    print(f"Active Tracks: {tracker.active_tracks}")
    print(f"Confirmed Tracks: {confirmed_tracks}")
    print(f"Target ID: {tracker.target_track_id}") # Should be ID 2 (larger box)

    print("\n--- Simulating Mouse Click Selection ---")
    # Simulate click on Track 0 (assuming its box is roughly [80, 80, 180, 180])
    # Need a dummy event, flags, param for the call
    class DummyEvent: pass
    tracker.handle_mouse_click(cv2.EVENT_LBUTTONDOWN, 100, 100, None, None)
    print(f"Selected ID: {tracker.selected_track_id}")
    print(f"Target ID: {tracker.target_track_id}") # Should now be ID 0

    print("\n--- Testing Relative Position ---")
    frame_width = 640
    relative_pos = tracker.get_target_relative_horizontal_position(frame_width)
    print(f"Target Box (ID {tracker.target_track_id}): {tracker.get_target_info()}")
    print(f"Relative Horizontal Position: {relative_pos:.2f}% (Frame Width: {frame_width})")

    # Simulate losing target
    print("\n--- Simulating Target Loss (No Detections) ---")
    for i in range(tracker.max_age + 1):
        print(f"--- Frame {11 + i} (No Detections) ---")
        tracker.update([])
        print(f"Active Tracks: {tracker.active_tracks}")
        print(f"Target ID: {tracker.target_track_id}")
        if tracker.target_track_id is None:
            print("Target lost as expected.")
            break

    print("\n✅ SimpleTracker test finished.") 