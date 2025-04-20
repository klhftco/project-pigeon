from djitellopy import Tello
import cv2
import time
import socket
import sys
import os

def release_port(port):
    """Try to release a port that's in use"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', port))
        sock.close()
        return True
    except socket.error:
        return False

def find_available_ports():
    """Find available ports for video and state"""
    # Default ports
    video_port = 8890
    state_port = 8891
    
    # Try to find available video port
    if not release_port(video_port):
        print(f"Could not release video port {video_port}. Trying alternatives...")
        alt_video_ports = [8892, 8893, 8894]
        for port in alt_video_ports:
            if release_port(port):
                video_port = port
                print(f"Using alternative video port {port}")
                break
        else:
            print("Could not find available video port")
            return None, None

    # Try to find available state port
    if not release_port(state_port):
        print(f"Could not release state port {state_port}. Trying alternatives...")
        alt_state_ports = [8895, 8896, 8897]
        for port in alt_state_ports:
            if release_port(port):
                state_port = port
                print(f"Using alternative state port {port}")
                break
        else:
            print("Could not find available state port")
            return None, None

    return video_port, state_port

def main():
    # Find available ports
    video_port, state_port = find_available_ports()
    if video_port is None or state_port is None:
        print("Could not find available ports. Please restart your computer.")
        sys.exit(1)

    # Initialize and connect to Tello with custom ports
    tello = Tello()
    tello.video_port = video_port
    tello.state_port = state_port
    
    try:
        print("Connecting to Tello...")
        tello.connect()
        print(f"Connected! Battery: {tello.get_battery()}%")

        # Start video stream
        print("Starting video stream...")
        tello.streamon()
        time.sleep(2)  # Give time for stream to start

        while True:
            try:
                # Read frame from the video stream
                frame = tello.get_frame_read().frame
                if frame is None:
                    print("No frame received from Tello")
                    continue

                # Display the frame
                cv2.imshow("Tello Live Stream", frame)

                # Quit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error in main loop: {e}")
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        print("Cleaning up...")
        try:
            tello.streamoff()
        except:
            pass
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()
