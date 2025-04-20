# Tello Drone Project

A Python-based project for controlling and tracking objects with a DJI Tello drone.

## Project Structure

- `drone_tracker/`: Main drone tracking module
  - `drone_controller.py`: Core controller for drone operations
  - `drone_controller_owl.py`: OWL-ViT based controller
  - `drone_tracker_face.py`: Face tracking implementation
  - `drone_tracker_face_owl.py`: OWL-ViT face tracking
  - `yolo_detector.py`: YOLO-based object detection
  - `owl_detector.py`: OWL-ViT object detection
  - `tracker.py`: Base tracking implementation
  - `yolov8n.pt`: YOLO model weights

- Core Scripts:
  - `app.py`: Main application interface
  - `tracking.py`: Object tracking algorithms
  - `vision.py`: Computer vision utilities
  - `record.py`: Manual flight recording tool
  - `tello_pydnet_interface.py`: Interface for collision avoidance

- Support Files:
  - `requirements.txt`: Project dependencies
  - `haarcascade_frontalface_default.xml`: Face detection model
  - `image_sample.jpg`: Sample image for testing

## Setup Instructions

1. Create and activate virtual environment:
```bash
conda create -n pje python=3.12
conda activate pje
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Additional system requirements:
```bash
brew install portaudio  # For audio support
```

## Running the Project

1. Basic drone tracking:
```bash
uv run python -m drone_tracker.drone_controller
```

2. Face tracking:
```bash
uv run python -m drone_tracker.drone_tracker_face
```

3. OWL-ViT tracking:
```bash
uv run python -m drone_tracker.drone_controller_owl
```

4. Manual flight recording:
```bash
python record.py
```

5. Web interface:
```bash
python app.py
```

## Features

- Object tracking using YOLO and OWL-ViT
- Face detection and tracking
- Predefined motion sequences
- Manual flight control
- Video streaming and recording
- Web interface for monitoring
- Collision avoidance capabilities

## Motion Sequences

The motion controller supports various predefined sequences:
- Takeoff and landing
- 360-degree rotations
- Custom circular patterns
- Height and speed control
- Face-following patterns

## Requirements

- Python 3.12
- DJI Tello drone
- WiFi connection
- OpenCV
- YOLO model
- OWL-ViT model
- PortAudio (for audio features)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
```