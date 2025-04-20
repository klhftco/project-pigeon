import torch
from ultralytics import YOLO
import os

class YoloDetector:
    """Handles loading the YOLO model and performing object detections."""

    def __init__(self, model_path="yolov8n.pt", device=None):
        """
        Initializes the YOLO detector.

        Args:
            model_path (str): Path to the YOLO model file.
            device (str, optional): Device to run the model on ('cpu', 'mps', 'cuda'). 
                                     Defaults to auto-detect MPS or fallback to CPU.
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
                print("✅ YoloDetector: Using MPS device.")
            # elif torch.cuda.is_available(): # Add check for CUDA if needed
            #     self.device = 'cuda'
            #     print("✅ YoloDetector: Using CUDA device.")
            else:
                self.device = 'cpu'
                print("ℹ️ YoloDetector: MPS/CUDA not available, using CPU.")
        else:
            self.device = device
            print(f"ℹ️ YoloDetector: Using specified device: {self.device}")

        # Use absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        abs_model_path = os.path.join(current_dir, model_path)
        print(f"📁 Loading model from: {abs_model_path}")
        self.model = YOLO(abs_model_path)
        print(f"✅ YoloDetector: Model '{model_path}' loaded successfully.")

        # Get the class ID for 'person'
        self.person_class_id = 0 # Default
        try:
            # Ensure model.names is accessible and is a dict-like object
            if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
                 self.person_class_id = list(self.model.names.keys())[list(self.model.names.values()).index('person')]
                 print(f"ℹ️ YoloDetector: Found 'person' class ID: {self.person_class_id}")
            else:
                 print("⚠️ YoloDetector: Model names not found or not in expected format. Using default class ID 0 for 'person'.")
                 # Attempt prediction to potentially populate names if lazy-loaded
                 _ = self.model.predict(torch.zeros(1, 3, 640, 640).to(self.device), verbose=False)
                 if hasattr(self.model, 'names') and isinstance(self.model.names, dict):
                      try:
                          self.person_class_id = list(self.model.names.keys())[list(self.model.names.values()).index('person')]
                          print(f"ℹ️ YoloDetector: Found 'person' class ID after prediction: {self.person_class_id}")
                      except ValueError:
                           print("⚠️ YoloDetector: 'person' class still not found after prediction. Using class ID 0.")
                 else:
                      print("⚠️ YoloDetector: Model names still unavailable after prediction. Using class ID 0.")

        except ValueError:
            print(f"⚠️ YoloDetector: 'person' class not found in model names. Using default class ID {self.person_class_id}.")
        except Exception as e:
            print(f"❌ YoloDetector: Error accessing model names: {e}. Using default class ID {self.person_class_id}.")


    def detect(self, frame, conf_threshold=0.30, verbose=False):
        """
        Performs object detection on a single frame.

        Args:
            frame (np.ndarray): The input image frame.
            conf_threshold (float): Confidence threshold for detections.
            verbose (bool): Whether to print YOLO prediction details.

        Returns:
            list: A list of detections for the 'person' class. 
                  Each detection is [[x1, y1, x2, y2], score].
        """
        results = self.model.predict(frame, conf=conf_threshold, verbose=verbose, device=self.device)[0]

        detections = []
        for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls) == self.person_class_id:
                detections.append([box.cpu().numpy().tolist(), float(score.cpu().numpy())]) # Ensure data is on CPU and standard types
        return detections

if __name__ == '__main__':
    # Example Usage (if you want to test this file directly)
    import cv2
    import time

    print("Testing YoloDetector...")
    detector = YoloDetector()
    cap = cv2.VideoCapture(0) # Use webcam

    if not cap.isOpened():
        print("❌ Error: Cannot open webcam.")
        exit()

    start_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Cannot read frame.")
            break

        detections = detector.detect(frame)
        frame_count += 1

        # Draw detections
        for det in detections:
            box, score = det
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate FPS
        if time.time() - start_time >= 1.0:
            fps = frame_count / (time.time() - start_time)
            start_time = time.time()
            frame_count = 0

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("YoloDetector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ YoloDetector test finished.") 