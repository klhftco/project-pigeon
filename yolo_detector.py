# yolo_detector.py
import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Get the 'person' class index (usually 0)
person_class_id = 0
try:
    person_class_id = list(model.names.keys())[list(model.names.values()).index('person')]
except Exception:
    print("Warning: 'person' class not found. Defaulting to class ID 0.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def findBoundingBox(frame, labels=["a person"], threshold=0.3):
    """
    frame: BGR OpenCV frame
    returns:
        frame: with drawn box
        info: [[cx, cy], area]
        label: "person" or None
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(frame, conf=threshold, verbose=False, device=str(device))[0]

    boxes = []
    for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == person_class_id and float(score) >= threshold:
            boxes.append((box.tolist(), float(score)))

    if not boxes:
        return frame, [[0, 0], 0], None

    # Select largest person box
    largest_box, best_score = max(boxes, key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]))
    x0, y0, x1, y1 = map(int, largest_box)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    area = (x1 - x0) * (y1 - y0)

    # Draw box and label
    label = "person"
    caption = f"{label}: {best_score:.2f}"
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(frame, caption, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return frame, [[cx, cy], area], label
