import cv2
import numpy as np
import threading
from queue import Queue
from ultralytics import YOLO
import time
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

class YOLOProcessor:
    def __init__(self, model_path='yolov8n.pt'):
        # Check for GPU availability
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model and move to device
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,  # Maximum number of missed misses before a track is deleted
            n_init=3,    # Number of frames that a track remains in initialization phase
            nms_max_overlap=1.0,  # Non-maxima suppression threshold
            max_cosine_distance=0.3,  # Gating threshold for cosine distance
            nn_budget=None,  # Maximum size of the appearance descriptors
            override_track_class=None,
            embedder="mobilenet",  # Feature extractor
            half=True,  # Use half precision
            bgr=True,  # Expect BGR images
            embedder_gpu=True  # Use GPU for feature extraction if available
        )
        
        self.input_queue = Queue(maxsize=2)  # Small queue to prevent lag
        self.output_queue = Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.process_thread = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0
        self.is_ready = False  # Flag to indicate if processor is ready
        
        # Optimized YOLO input size (smaller but still effective)
        self.yolo_width = 640  # Reduced from 736
        self.yolo_height = 640  # Reduced from 736
        
        # Frame skipping for better performance
        self.frame_skip = 1  # Process every other frame
        self.frame_counter = 0

    def start_processing(self):
        """Start the YOLO processing thread"""
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        return self

    def _process_loop(self):
        """Main processing loop"""
        while not self.stop_event.is_set():
            try:
                # Get frame from input queue
                frame = self.input_queue.get(timeout=0.1)
                if frame is None:
                    continue

                # Skip frames for better performance
                self.frame_counter += 1
                if self.frame_counter % (self.frame_skip + 1) != 0:
                    continue

                # Process frame with YOLO
                start_time = time.time()
                results = self.model(frame, 
                                   classes=[0],  # Only detect people
                                   imgsz=self.yolo_height,
                                   conf=0.5,  # Higher confidence threshold
                                   verbose=False,  # Disable verbose output
                                   device=self.device)  # Use specified device
                
                # Convert YOLO detections to DeepSORT format
                detections = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = box.conf[0].item()
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
                
                # Update tracker
                tracks = self.tracker.update_tracks(detections, frame=frame)
                
                # Draw tracked objects
                processed_frame = frame.copy()
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    
                    x1, y1, x2, y2 = map(int, ltrb)
                    
                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw track ID
                    cv2.putText(processed_frame, f'ID: {track_id}', (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_frame_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_frame_time = current_time

                # Add FPS to frame
                cv2.putText(processed_frame, f'YOLO FPS: {self.fps}', (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Put processed frame in output queue
                if self.output_queue.full():
                    self.output_queue.get_nowait()
                self.output_queue.put_nowait(processed_frame)

            except Exception as e:
                if self.is_ready:  # Only print errors if processor is ready
                    print(f"Processing error: {e}")
                time.sleep(0.1)

    def put_frame(self, frame):
        """Add a frame to the processing queue"""
        if frame is not None:
            if self.input_queue.full():
                self.input_queue.get_nowait()
            self.input_queue.put_nowait(frame)
            self.is_ready = True  # Mark processor as ready after first frame

    def get_processed_frame(self):
        """Get the latest processed frame"""
        try:
            return self.output_queue.get_nowait()
        except:
            return None

    def stop(self):
        """Stop the processing thread"""
        self.stop_event.set()
        if self.process_thread:
            self.process_thread.join()

def main():
    from stream import TelloStream
    
    # Initialize stream and YOLO processor
    stream = TelloStream()
    yolo = YOLOProcessor().start_processing()
    
    if not stream.connect():
        print("Failed to connect to Tello")
        return

    try:
        # Wait for stream to stabilize
        time.sleep(2)
        
        while True:
            # Get frame from stream
            frame = stream.get_frame()
            if frame is not None:
                # Send frame to YOLO processor
                yolo.put_frame(frame)
                
                # Get processed frame
                processed_frame = yolo.get_processed_frame()
                if processed_frame is not None:
                    cv2.imshow("Tello YOLO", processed_frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        yolo.stop()
        stream.cleanup()

if __name__ == "__main__":
    main() 