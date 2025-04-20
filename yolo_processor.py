import cv2
import numpy as np
import threading
from queue import Queue
from ultralytics import YOLO
import time
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import psutil
import multiprocessing

class YOLOProcessor:
    def __init__(self, model_path='yolov8n.pt'):
        # CPU-only configuration with increased resource usage
        self.device = 'cpu'
        print("Running on CPU (Intel Mac)")
        
        # Get number of CPU cores and use most of them
        self.num_cores = multiprocessing.cpu_count()
        self.threads = max(1, self.num_cores - 1)  # Leave one core free for system
        print(f"Using {self.threads} CPU cores")
        
        # Set OpenCV to use multiple threads
        cv2.setNumThreads(self.threads)
        
        try:
            # Load model optimized for CPU with increased batch size
            self.model = YOLO(model_path)
            self.model.to(self.device)
            # Set model to use more threads
            torch.set_num_threads(self.threads)
            print("Model loaded successfully on CPU")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        try:
            # Initialize DeepSORT tracker with CPU-optimized settings
            self.tracker = DeepSort(
                max_age=15,
                n_init=2,
                nms_max_overlap=1.0,
                max_cosine_distance=0.4,
                nn_budget=100,  # Increased feature budget
                override_track_class=None,
                embedder="mobilenet",
                half=False,
                bgr=True,
                embedder_gpu=False
            )
            print("DeepSORT tracker initialized successfully")
        except Exception as e:
            print(f"Error initializing tracker: {e}")
            raise
        
        # Optimize queue sizes for better throughput
        self.input_queue = Queue(maxsize=3)  # Increased queue size
        self.output_queue = Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.process_thread = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0
        self.is_ready = False
        
        # Optimized YOLO input size for better CPU utilization
        self.yolo_width = 640  # Increased resolution for better accuracy
        self.yolo_height = 640
        
        # Reduced frame skip for better tracking
        self.frame_skip = 1  # Process every other frame
        self.frame_counter = 0
        self.tracking_interval = 2  # More frequent tracking updates
        
        # Performance monitoring
        self.last_perf_check = time.time()
        self.perf_check_interval = 2.0

    def _get_performance_info(self):
        """Get performance information"""
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            
            # Get system-wide CPU usage
            system_cpu = psutil.cpu_percent(interval=0.1)
            
            return f"CPU: {cpu_percent}% (System: {system_cpu}%) | Memory: {memory_info.rss / 1024 / 1024:.1f}MB | FPS: {self.fps}"
        except Exception as e:
            return f"Monitoring error: {e}"

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

                # Check and display performance info periodically
                current_time = time.time()
                if current_time - self.last_perf_check >= self.perf_check_interval:
                    print(f"\r{self._get_performance_info()}", end="")
                    self.last_perf_check = current_time

                # Process frame with YOLO
                try:
                    # Resize frame for better CPU performance
                    resized_frame = cv2.resize(frame, (self.yolo_width, self.yolo_height))
                    
                    # Process with increased batch size
                    results = self.model(resized_frame, 
                                       classes=[0],
                                       imgsz=self.yolo_height,
                                       conf=0.5,
                                       verbose=False,
                                       device=self.device,
                                       batch=1)  # Process one frame at a time
                    
                    # Convert YOLO detections to DeepSORT format
                    detections = []
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        # Scale coordinates back to original frame size
                        x1 = int(x1 * frame.shape[1] / self.yolo_width)
                        y1 = int(y1 * frame.shape[0] / self.yolo_height)
                        x2 = int(x2 * frame.shape[1] / self.yolo_width)
                        y2 = int(y2 * frame.shape[0] / self.yolo_height)
                        conf = box.conf[0].item()
                        detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
                    
                    # Only run tracking every N frames
                    if self.frame_counter % self.tracking_interval == 0:
                        tracks = self.tracker.update_tracks(detections, frame=frame)
                    else:
                        tracks = self.tracker.tracks
                    
                    # Draw tracked objects
                    processed_frame = frame.copy()
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        
                        track_id = track.track_id
                        ltrb = track.to_ltrb()
                        
                        x1, y1, x2, y2 = map(int, ltrb)
                        
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(processed_frame, f'ID: {track_id}', (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Update FPS
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
                    print(f"\nProcessing error: {e}")
                    time.sleep(0.1)

            except Exception as e:
                if self.is_ready:
                    print(f"\nQueue error: {e}")
                time.sleep(0.1)

    def put_frame(self, frame):
        """Add a frame to the processing queue"""
        if frame is not None:
            if self.input_queue.full():
                self.input_queue.get_nowait()
            self.input_queue.put_nowait(frame)
            self.is_ready = True

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