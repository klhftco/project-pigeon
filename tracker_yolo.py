#!/usr/bin/env python3
"""
tracker_yolo.py — minimal YOLO demo with simple IoU tracker.
Detects only people from webcam 0 and assigns track IDs.
"""
import sys, cv2, time
import numpy as np # Add numpy for IoU calculation
from ultralytics import YOLO
import torch

# --- Tracker Constants ---
IOU_THRESHOLD = 0.4  # Minimum IoU to match a detection to a track
MAX_AGE = 5          # Max frames a track can exist without detection
MIN_HITS = 3         # Min detections needed to confirm a track
# -------------------------

# --- Global State for Interaction ---
active_tracks = [] # Global list to store active tracks (shared with callback)
selected_track_id = None # ID of the track selected by clicking
target_track_id = None   # ID of the track automatically targeted
# ----------------------------------

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

# --- Mouse Callback for Track Selection ---
def select_track(event, x, y, flags, param):
    global selected_track_id, active_tracks, MIN_HITS

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_track_id = None # Reset selection on new click
        clicked_tracks = []

        for track in active_tracks:
            if track['hits'] >= MIN_HITS: # Consider only confirmed tracks
                x0, y0, x1, y1 = map(int, track['box'])
                if x0 <= x <= x1 and y0 <= y <= y1:
                    area = (x1 - x0) * (y1 - y0)
                    clicked_tracks.append({'id': track['id'], 'area': area, 'track': track})

        if clicked_tracks:
            # Sort overlapping tracks by area (smallest first)
            clicked_tracks.sort(key=lambda t: t['area'])
            smallest_track = clicked_tracks[0]
            selected_track_id = smallest_track['id']
            print(f"🖱️ Click selected Track ID: {selected_track_id}")
            print(f"   Track details: {smallest_track['track']}")
# ----------------------------------------

def main():
    global active_tracks, next_track_id, selected_track_id, target_track_id, MIN_HITS # Use global state

    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✅ Using MPS device.")
    else:
        device = 'cpu'
        print("ℹ️ MPS not available, using CPU.")

    model = YOLO("yolov8s.pt") # Load standard small YOLOv8 model

    # We will filter for the 'person' class ID later.
    person_class_id = 0 # COCO dataset usually has 'person' as class ID 0
    try:
        person_class_id = list(model.names.keys())[list(model.names.values()).index('person')]
    except ValueError:
        print("Warning: 'person' class not found in model names. Using class ID 0.")
        person_class_id = 0 # Default fallback

    # Default to webcam 0
    source = 0
    conf_threshold = 0.30

    cap = cv2.VideoCapture(source)
    if not cap or not cap.isOpened():
        print(f"❌ Error opening video source: {source}")
        sys.exit(1)

    last_fps_update_time = time.time()
    frame_count_total = 0 # Renamed to avoid conflict with FPS frame_count
    frame_count_fps = 0   # Use separate counter for FPS calculation
    display_fps = 0.0

    # --- Tracker State ---
    # active_tracks = [] # Now global
    next_track_id = 0
    # --------------------

    window_name = "YOLOv8 - Person Tracking"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_track) # Register the callback

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌  End of video stream or cannot read frame.")
            break

        # Predict
        results = model.predict(frame, conf=conf_threshold, verbose=False, device=device)[0]
        img_to_draw_on = frame.copy() # Draw on a copy to avoid modifying original frame used by tracker potentially

        # --- Prepare Detections for Tracker ---
        current_detections = [] # Format: [[x1, y1, x2, y2], score]
        for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls) == person_class_id:
                current_detections.append([box.tolist(), float(score)])
        # ------------------------------------

        # --- Tracker Update Logic ---
        matched_indices = set()
        track_indices = list(range(len(active_tracks)))
        detection_indices = list(range(len(current_detections)))

        if not active_tracks or not current_detections:
            # No tracks or no detections, skip matching
            pass
        else:
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(active_tracks), len(current_detections)))
            for t_idx, track in enumerate(active_tracks):
                for d_idx, detection in enumerate(current_detections):
                    iou_matrix[t_idx, d_idx] = calculate_iou(track['box'], detection[0])

            # Simple greedy matching based on highest IoU
            # More sophisticated methods like Hungarian algorithm exist but are more complex
            for _ in range(min(len(active_tracks), len(current_detections))):
                if np.max(iou_matrix) < IOU_THRESHOLD:
                    break # No more good matches

                # Find best match
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)

                # Mark as matched and update track
                track = active_tracks[t_idx]
                detection_box = current_detections[d_idx][0]
                track['box'] = detection_box
                track['age'] = 0 # Reset age on match
                track['hits'] += 1

                matched_indices.add(d_idx)
                track_indices.remove(t_idx)
                # Prevent rematching
                iou_matrix[t_idx, :] = -1
                iou_matrix[:, d_idx] = -1

        # Handle unmatched tracks (increment age or remove)
        new_active_tracks = []
        for t_idx in range(len(active_tracks)):
            track = active_tracks[t_idx]
            if t_idx not in track_indices: # If it was matched, keep it
                 new_active_tracks.append(track)
                 continue
            # Unmatched: increment age and check if too old
            track['age'] += 1
            if track['age'] <= MAX_AGE:
                new_active_tracks.append(track)
        active_tracks = new_active_tracks

        # Handle unmatched detections (create new tracks)
        for d_idx in range(len(current_detections)):
            if d_idx not in matched_indices:
                new_track = {
                    'id': next_track_id,
                    'box': current_detections[d_idx][0],
                    'age': 0,
                    'hits': 1
                }
                active_tracks.append(new_track)
                next_track_id += 1
        # --------------------------

        # --- Auto Target Selection (at frame 10) ---
        frame_count_total += 1
        if frame_count_total == 10 and target_track_id is None:
            confirmed_tracks = [t for t in active_tracks if t['hits'] >= MIN_HITS]
            if confirmed_tracks:
                largest_track = max(confirmed_tracks, key=lambda t: (t['box'][2]-t['box'][0])*(t['box'][3]-t['box'][1]))
                target_track_id = largest_track['id']
                print(f"🎯 Auto-selected largest track as target: ID {target_track_id}")
                print(f"   Target track details: {largest_track}")
        # -----------------------------------------

        # --- Drawing Logic ---
        # Draw confirmed tracks with IDs
        for track in active_tracks:
            if track['hits'] >= MIN_HITS: # Only draw confirmed tracks
                box = list(map(int, track['box']))
                track_id = track['id']
                x0, y0, x1, y1 = box
                # Use track ID for color variation (simple approach)
                color = ((track_id * 30) % 255, (track_id * 50) % 255, (track_id * 70) % 255)
                thickness = 2
                label = f"ID:{track_id}"

                # Highlight selected and target tracks
                if track_id == selected_track_id:
                    color = (0, 255, 255) # Bright Yellow
                    thickness = 3
                    label = f"ID:{track_id} [SELECTED]"
                elif track_id == target_track_id:
                    color = (255, 255, 0) # Bright Cyan
                    thickness = 3
                    label = f"ID:{track_id} [TARGET]"

                cv2.rectangle(img_to_draw_on, (x0, y0), (x1, y1), color, thickness)
                cv2.putText(img_to_draw_on, label, (x0, y0 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        # ---------------------

        # Calculate and display FPS (smoothed over 1 second)
        frame_count_fps += 1
        current_time = time.time()
        elapsed_time = current_time - last_fps_update_time

        if elapsed_time >= 1.0:
            display_fps = frame_count_fps / elapsed_time
            frame_count_fps = 0
            last_fps_update_time = current_time

        cv2.putText(img_to_draw_on, f"FPS: {display_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Changed color for visibility

        # Display
        cv2.imshow(window_name, img_to_draw_on) # Use named window

        # Wait for key press (1ms delay)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):  # ESC or q to quit
            break

    # Cleanup
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 