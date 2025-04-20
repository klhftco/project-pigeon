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

        # OWL-ViT uses text prompts, so we define the target classes here
        # Multiple prompts for better color/clothing detection
        self.target_texts = [
            # "a person wearing a hat",
            # "a hat",
            # "a baseball cap",
            # "a person wearing a baseball cap",
            # "a man wearing a hat",
            # "a woman wearing a hat",
            "a person wearing blue",
            "a person wearing a blue shirt",
            "a person wearing a blue t-shirt"
        ]

    def _enhance_colors(self, bgr_frame):
        """
        Enhance colors in the frame to improve detection of colored objects.

        Args:
            bgr_frame (np.ndarray): BGR format frame from OpenCV

        Returns:
            np.ndarray: Enhanced BGR frame
        """
        try:
            # Auto white-balance using Grayworld algorithm
            if hasattr(cv2, 'xphoto') and hasattr(cv2.xphoto, 'createGrayworldWB'):
                bgr_frame = cv2.xphoto.createGrayworldWB().balanceWhite(bgr_frame)
            else:
                print("⚠️ cv2.xphoto not available - skipping white balance")

            # Light denoise (ISO speckle kills color info)
            bgr_frame = cv2.fastNlMeansDenoisingColored(bgr_frame, None, 10, 10, 7, 21)

            # Slight gamma lift for better color visibility
            bgr_frame = cv2.convertScaleAbs(bgr_frame, alpha=1.3, beta=0)

            return bgr_frame

        except Exception as e:
            print(f"⚠️ Color enhancement failed: {e}, using original frame")
            return bgr_frame

    @torch.no_grad()
    def detect(self, frame, conf_threshold=0.30, verbose=False, target_texts=None):
        """
        Performs object detection on a single frame for the target texts.

        Args:
            frame (np.ndarray): The input image frame (in BGR format from OpenCV).
            conf_threshold (float): Confidence threshold for detections.
            verbose (bool): If True, prints transformer-specific warnings.
            target_texts (list, optional): Override default target texts with custom prompts.

        Returns:
            list: A list of detections.
                  Each detection is [[x1, y1, x2, y2], score].
        """
        # Use provided target_texts if specified, otherwise use default
        if target_texts is None:
            target_texts = self.target_texts

        # 0. Enhance colors to improve detection
        enhanced_frame = self._enhance_colors(frame.copy())

        # 1. Preprocess: Convert OpenCV BGR frame to PIL RGB Image
        image = Image.fromarray(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))

        # 2. Prepare inputs
        inputs = self.processor(text=target_texts, images=image, return_tensors="pt")

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

        # DEBUGGING: Print top confidence scores for each prompt
        logits = outputs.logits  # [B, num_queries, #prompts]
        probs = logits.sigmoid()[0].cpu().numpy()  # (num_queries, #prompts)

        print("top 5 scores for each prompt:")
        for p_idx, prompt in enumerate(target_texts):
            top = sorted(probs[:, p_idx], reverse=True)[:5]
            print(f"{prompt:<35}  {top}")

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
        # Process all detections across all prompts
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            # label_idx is the index into target_texts
            # Box coordinates are [x_min, y_min, x_max, y_max]
            box_coords = box.cpu().numpy().tolist()
            score_val = float(score.cpu().numpy())
            prompt_used = target_texts[label_idx]
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

        # Try with a lower threshold for testing
        detections = detector.detect(frame, conf_threshold=0.05)
        frame_count += 1

        # Draw detections
        for det in detections:
            box, score = det
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange box
            # For display, just show "Person" with score
            cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

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
