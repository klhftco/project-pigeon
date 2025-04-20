import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import numpy as np
import cv2 # For testing block
import time # For testing block

class OwlDetector:
    """Handles loading the OWL-ViT model and performing object detections."""

    def __init__(self, model_path="google/owlvit-base-patch32", device=None):
        """
        Initializes the OWL-ViT detector.

        Args:
            model_path (str): Identifier for the pretrained OWL-ViT model.
            device (str, optional): Device to run the model on ('cpu', 'mps', 'cuda').
                                     Defaults to auto-detect CUDA, MPS or fallback to CPU.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("✅ OwlDetector: Using CUDA device.")
            elif torch.backends.mps.is_available():
                # Check if MPS is truly functional
                try:
                    # Attempt a simple operation on MPS device
                    _ = torch.tensor([1.0], device="mps") * 2.0
                    self.device = torch.device('mps')
                    print("✅ OwlDetector: Using MPS device.")
                except Exception as e:
                    print(f"⚠️ OwlDetector: MPS available but failed test ({e}), falling back to CPU.")
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cpu')
                print("ℹ️ OwlDetector: CUDA/MPS not available, using CPU.")
        else:
            self.device = torch.device(device)
            print(f"ℹ️ OwlDetector: Using specified device: {self.device}")

        print(f"ℹ️ OwlDetector: Loading model '{model_path}'...")
        self.processor = OwlViTProcessor.from_pretrained(model_path)
        self.model = OwlViTForObjectDetection.from_pretrained(model_path).to(self.device)

        # Convert model to half precision if using MPS for potentially better performance
        if self.device.type == "mps":
            # Check if half precision is supported and beneficial
            try:
                # Basic check if the model seems compatible
                if hasattr(self.model, 'half'):
                    self.model.half()
                    print("✅ OwlDetector: Model converted to half precision (fp16) for MPS.")
                else:
                    print("⚠️ OwlDetector: Model does not have `half()` method, cannot convert for MPS.")
            except Exception as e:
                print(f"⚠️ OwlDetector: Failed to convert model to half precision for MPS: {e}")


        self.model.eval()
        print(f"✅ OwlDetector: Model '{model_path}' loaded successfully on {self.device}.")

        # OWL-ViT uses text prompts, so we define the target class here.
        # To be a drop-in replacement for YoloDetector, we'll hardcode 'person'.
        self.target_texts = ["a person"]


    @torch.no_grad()
    def detect(self, frame, conf_threshold=0.30, verbose=False):
        """
        Performs object detection on a single frame for the target text "a person".

        Args:
            frame (np.ndarray): The input image frame (in BGR format from OpenCV).
            conf_threshold (float): Confidence threshold for detections.
            verbose (bool): If True, prints transformer-specific warnings (currently unused).

        Returns:
            list: A list of detections for 'person'.
                  Each detection is [[x1, y1, x2, y2], score].
        """
        # 1. Preprocess: Convert OpenCV BGR frame to PIL RGB Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 2. Prepare inputs
        inputs = self.processor(text=self.target_texts, images=image, return_tensors="pt")

        # Move inputs to the correct device and handle MPS half precision
        inputs_on_device = {}
        for k, v in inputs.items():
            if self.device.type == "mps" and v.dtype.is_floating_point:
                 # Check if model is also in half precision before converting input
                 is_model_half = next(self.model.parameters()).dtype == torch.float16
                 if is_model_half:
                     inputs_on_device[k] = v.to(self.device).half()
                 else: # Don't convert input if model isn't half
                     inputs_on_device[k] = v.to(self.device)
            else:
                inputs_on_device[k] = v.to(self.device)
        inputs = inputs_on_device


        # 3. Inference
        outputs = self.model(**inputs)

        # 4. Post-process
        # Target sizes must be (h, w); PIL size is (w, h)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)

        # Check if model is half precision for potential post-processing adjustments
        is_model_half = next(self.model.parameters()).dtype == torch.float16
        if is_model_half and self.device.type == "mps":
             # Potentially cast logits to float32 if needed by post_process
             if hasattr(outputs, 'logits') and outputs.logits.dtype == torch.float16:
                 outputs.logits = outputs.logits.float()
             if hasattr(outputs, 'pred_boxes') and outputs.pred_boxes.dtype == torch.float16:
                 outputs.pred_boxes = outputs.pred_boxes.float()

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=conf_threshold,
        )[0]  # Get the results for the first (and only) image

        # 5. Format results to match YoloDetector output
        detections = []
        # OWL-ViT returns labels as indices into the input text list.
        # Since we only provide "a person", the label index should always be 0.
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            if label_idx == 0: # Corresponds to "a person"
                 # Box coordinates are [x_min, y_min, x_max, y_max]
                 box_coords = box.cpu().numpy().tolist()
                 score_val = float(score.cpu().numpy())
                 detections.append([box_coords, score_val])

        return detections

if __name__ == '__main__':
    # Example Usage (similar to YoloDetector test)
    print("Testing OwlDetector...")
    # Use default model and auto-detect device
    detector = OwlDetector(conf_threshold=0.2) # Lower threshold might be needed for OWL-ViT
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
            label = detector.target_texts[0] # Get the text label ("a person")
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate FPS
        current_time = time.time()
        if current_time - start_time >= 1.0:
            fps = frame_count / (current_time - start_time)
            start_time = current_time
            frame_count = 0

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("OwlDetector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ OwlDetector test finished.") 