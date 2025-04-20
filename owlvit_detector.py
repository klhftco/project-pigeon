# owlvit_detector.py
import torch
import cv2
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
model.eval()

@torch.no_grad()
def detect(
    model: OwlViTForObjectDetection,
    processor: OwlViTProcessor,
    image: Image.Image,
    texts: list[str],
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

@torch.no_grad()
def findBoundingBox(frame, labels=["a person"], threshold=0.3):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    # Inference & post‑processing
    detections = detect(
        model,
        processor,
        image,
        texts=labels,
        threshold=threshold,
        device=device,
    )

    # Draw boxes & labels
    boxes = detections["boxes"].cpu()
    scores = detections["scores"].cpu()
    label_idxs = detections["labels"].cpu()  # indices into args.labels

    if len(boxes) == 0 or scores[0] < threshold:
        return frame, [[0, 0], 0], None

    # Get first box
    x0, y0, x1, y1 = map(int, boxes[0].tolist())
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    area = (x1 - x0) * (y1 - y0)
    label = labels[label_idxs[0]]

    # Draw box
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    caption = f"{label}: {scores[0]:.2f}"
    cv2.putText(frame, caption, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame, [[cx, cy], area], label
