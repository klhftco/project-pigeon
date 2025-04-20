import cv2
import numpy as np
import subprocess
import time
import threading
from queue import Queue
from djitellopy import Tello
import os
import signal

class TelloStream:
    def __init__(self):
        self.tello = Tello()
        self.width, self.height = 960, 720  # Tello's actual resolution
        self.frame_size = self.width * self.height * 3
        self.frame_queue = Queue(maxsize=2)  # Small queue to prevent lag
        self.stop_event = threading.Event()
        self.stream_thread = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0
        self.ffmpeg_process = None
        self.cap = None

    def _kill_ffmpeg(self):
        """Safely kill FFmpeg process"""
        if self.ffmpeg_process:
            try:
                # Try graceful termination first
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=1)
            except:
                try:
                    # Force kill if needed
                    os.kill(self.ffmpeg_process.pid, signal.SIGKILL)
                except:
                    pass
            finally:
                self.ffmpeg_process = None

    def connect(self):
        """Connect to the Tello drone and start video stream"""
        try:
            print("Connecting to Tello...")
            self.tello.connect()
            
            print("Starting video stream...")
            self.tello.streamon()
            time.sleep(2)  # Give time for stream to initialize
            
            # Try multiple streaming methods
            if not self._start_ffmpeg_stream():
                print("FFmpeg stream failed, trying OpenCV...")
                if not self._start_opencv_stream():
                    print("All streaming methods failed")
                    return False
            
            print("Stream started successfully!")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def _start_ffmpeg_stream(self):
        """Start FFmpeg process with optimized settings"""
        try:
            # Kill any existing FFmpeg process
            self._kill_ffmpeg()

            # FFmpeg command with optimized settings
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", "udp://0.0.0.0:11111",
                "-pix_fmt", "bgr24",
                "-f", "rawvideo",
                "-r", "30",  # Force 30fps
                "-probesize", "32",  # Faster stream detection
                "-analyzeduration", "0",  # No stream analysis delay
                "-fflags", "nobuffer",  # Reduce latency
                "-flags", "low_delay",  # Reduce latency
                "-strict", "experimental",  # Allow experimental features
                "-"
            ]

            # Start FFmpeg process with proper cleanup
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,  # Larger buffer
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Start the stream thread
            self.stream_thread = threading.Thread(target=self._stream_loop)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            return True
        except Exception as e:
            print(f"FFmpeg error: {e}")
            self._kill_ffmpeg()
            return False

    def _start_opencv_stream(self):
        """Fallback to OpenCV stream if FFmpeg fails"""
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture("udp://0.0.0.0:11111")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            return self.cap.isOpened()
        except Exception as e:
            print(f"OpenCV error: {e}")
            return False

    def _stream_loop(self):
        """Main streaming loop"""
        while not self.stop_event.is_set():
            try:
                if self.ffmpeg_process:
                    # Read from FFmpeg
                    raw_frame = self.ffmpeg_process.stdout.read(self.frame_size)
                    if not raw_frame:
                        print("FFmpeg stream ended, restarting...")
                        self._kill_ffmpeg()
                        if not self._start_ffmpeg_stream():
                            time.sleep(1)
                            continue
                        continue
                    
                    frame = np.frombuffer(raw_frame, np.uint8).reshape((self.height, self.width, 3))
                elif self.cap and self.cap.isOpened():
                    # Read from OpenCV
                    ret, frame = self.cap.read()
                    if not ret:
                        print("OpenCV stream ended, restarting...")
                        self._start_opencv_stream()
                        continue
                else:
                    print("No valid stream source")
                    time.sleep(0.1)
                    continue

                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_frame_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_frame_time = current_time

                # Add frame to queue
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)

            except Exception as e:
                print(f"Stream error: {e}")
                time.sleep(0.1)

    def get_frame(self):
        """Get the latest frame from the queue"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return None

    def display_frame(self, frame):
        """Display frame with FPS counter"""
        if frame is not None:
            try:
                # Create a writable copy of the frame
                display_frame = frame.copy()
                cv2.putText(display_frame, f'FPS: {self.fps}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Tello Stream", display_frame)
            except Exception as e:
                print(f"Display error: {e}")

    def cleanup(self):
        """Clean up resources"""
        self.stop_event.set()
        
        if self.stream_thread:
            self.stream_thread.join()
            
        self._kill_ffmpeg()
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        self.tello.streamoff()
        cv2.destroyAllWindows()

def main():
    stream = TelloStream()
    
    if not stream.connect():
        print("Failed to connect to Tello")
        return

    try:
        while True:
            frame = stream.get_frame()
            if frame is not None:
                stream.display_frame(frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        stream.cleanup()

if __name__ == "__main__":
    main()
