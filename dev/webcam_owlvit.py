#!/usr/bin/env python3
"""
webcam_owlvit.py
----------------
Real‑time zero‑shot object detection on webcam frames with OWL‑ViT.

Dependencies (install with pip **after** Python 3.8+ is available):

    pip install transformers torch pillow opencv-python

Tested with:
    • transformers 4.39+
    • torch 2.2+
    • opencv‑python 4.10+
    • pillow 10+

Example usage:
    python webcam_owlvit.py                               # default "a person"
    python webcam_owlvit.py --labels "a cat" "a dog"
    python webcam_owlvit.py --labels "a bicycle" --threshold 0.25
"""
import argparse
from pathlib import Path
from typing import List
import time
import select # Add select
import sys # Add sys
import termios # Add termios for terminal settings
import tty # Add tty for terminal settings

import cv2                                   # webcam capture & drawing
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
)                                            # zero‑shot detector
from PIL import Image
import torch


def parse_args() -> argparse.Namespace:
    """Command‑line options."""
    parser = argparse.ArgumentParser(
        description="Real‑time zero‑shot object detection with OWL‑ViT."
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["chair"],
        help='Text queries (e.g. --labels "a cat" "a person").',
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Confidence threshold for drawing detections.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        choices=["cpu", "cuda", "mps"],
        help="Torch device to run the model on.",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="OpenCV camera index (default 0).",
    )
    return parser.parse_args()


@torch.no_grad()
def detect(
    model: OwlViTForObjectDetection,
    processor: OwlViTProcessor,
    image: Image.Image,
    texts: List[str],
    threshold: float,
    device: torch.device,
):
    """
    Run OWL‑ViT on a single PIL image and return post‑processed detections.
    """
    inputs = processor(text=texts, images=image, return_tensors="pt") # Keep inputs on CPU initially

    # Move inputs to device and convert to half precision if device is MPS
    if device.type == "mps":
        # Only convert float tensors to half precision
        inputs_on_device = {}
        for k, v in inputs.items():
            if v.dtype.is_floating_point:
                inputs_on_device[k] = v.to(device).half()
            else: # Keep integer tensors (like input_ids) as they are
                inputs_on_device[k] = v.to(device)
        inputs = inputs_on_device
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    # Target sizes must be (h, w); PIL size is (w, h)
    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )[0]  # list of length 1 → dict
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Store original terminal settings
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # Set terminal to raw mode for non-blocking character reading
        tty.setraw(sys.stdin.fileno())

        # ---------------- Model & processor ----------------
        print("Loading OWL‑ViT model …")
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(
            device
        )
        # Convert model to half precision if using MPS
        if device.type == "mps":
            model.half()
            print("Model converted to half precision (fp16) for MPS.")

        model.eval()
        print(f"Model loaded on {device}\r") # Use \r to allow overwriting

        # ---------------- Webcam ----------------
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            # Restore terminal settings before raising error
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            raise RuntimeError(f"Could not open webcam (index {args.camera_id})")

        print("\r                                                                   ", end="") # Clear previous line
        print(
            "🟢 Press 'q' in the display window to quit."
            f" Initial labels: {', '.join(args.labels)} (threshold {args.threshold})\r"
        )
        print("\nEnter new labels (comma-separated), then press Enter: ", end="", flush=True)

        # FPS calculation variables
        start_time = time.monotonic()
        frame_count = 0
        input_buffer = ""

        while True:
            # Check for keyboard input from console (non-blocking)
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                char = sys.stdin.read(1)
                if char == '\r' or char == '\n': # Enter key pressed
                    if input_buffer:
                        new_labels = [label.strip() for label in input_buffer.split(',') if label.strip()]
                        if new_labels:
                            args.labels = new_labels
                            print(f"\rLabels updated to: {', '.join(args.labels)}                        \r") # Overwrite previous prompt/input
                            # Reset frame count for FPS calculation if needed
                            frame_count = 0
                            start_time = time.monotonic()
                        else:
                            print("\r⚠️ No valid labels entered.                     \r") # Overwrite
                        input_buffer = ""
                    # Reprint prompt after processing or if buffer was empty
                    print("\nEnter new labels (comma-separated), then press Enter: ", end="", flush=True)
                elif char == '\x7f' or char == '\b': # Backspace key
                    if input_buffer:
                        input_buffer = input_buffer[:-1]
                        # Use ANSI escape code to move cursor back, print space, move back again
                        print("\b \b", end="", flush=True)
                elif char.isprintable():
                    input_buffer += char
                    print(char, end="", flush=True) # Echo printable characters
                elif char == '\x03': # Ctrl+C
                    print("\rCtrl+C detected, exiting.")
                    break

            ret, frame = cap.read()
            if not ret:
                print("\r⚠️  Failed to read frame — exiting.                    ")
                break

            # FPS calculation
            frame_count += 1
            elapsed_time = time.monotonic() - start_time
            if elapsed_time > 1: # Update FPS approx every second
                fps = frame_count / elapsed_time
                # print(f"FPS: {fps:.1f}") # Optional: print FPS if needed
                frame_count = 0
                start_time = time.monotonic()

            # Crop frame to 960x720 (bottom, centered)
            h, w = frame.shape[:2]
            target_w, target_h = 960, 720
            if h < target_h or w < target_w:
                # Restore terminal settings before raising error or skipping
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                print(f"\r⚠️ Frame ({w}x{h}) too small for {target_w}x{target_h} crop. Exiting.")
                # Or optionally, print a warning and continue:
                # print(f"\r⚠️ Frame ({w}x{h}) too small for {target_w}x{target_h} crop. Skipping frame.", end="", flush=True)
                # continue
                break # Exit if frame is too small
            y_start = h - target_h
            x_start = (w - target_w) // 2
            frame = frame[y_start:h, x_start:x_start + target_w]

            # Convert to PIL (RGB)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Inference & post‑processing
            detections = detect(
                model,
                processor,
                image,
                texts=args.labels,
                threshold=args.threshold,
                device=device,
            )

            # Draw boxes & labels
            boxes = detections["boxes"].cpu()
            scores = detections["scores"].cpu()
            labels = detections["labels"].cpu()  # indices into args.labels

            for box, score, label_idx in zip(boxes, scores, labels):
                x0, y0, x1, y1 = map(int, box.tolist())
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                caption = f"{args.labels[label_idx]}: {score:.2f}"
                cv2.putText(
                    frame,
                    caption,
                    (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Display
            cv2.imshow("OWL‑ViT Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\r'q' pressed in window, exiting.                       ")
                break

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam closed. Goodbye!")


if __name__ == "__main__":
    main()
