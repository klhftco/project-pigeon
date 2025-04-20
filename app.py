# app.py
from flask import Flask, render_template, request, jsonify, send_file
from openai import OpenAI
import json
import os
import time
import threading
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Import Tello controller classes
try:
    from djitellopy import Tello
    TELLO_AVAILABLE = True
    # from drone_tracker.drone_controller import DroneController
    from drone_tracker.drone_tracker_face_owl import DroneFaceTrackerOwl
except ImportError:
    print("WARNING: djitellopy not found. Drone functionality will be simulated.")
    TELLO_AVAILABLE = False

# Configuration
OFFLINE_MODE = True  # Set to True for offline mode without OpenAI API

app = Flask(__name__)

# Initialize OpenAI client if in online mode
client = None
if not OFFLINE_MODE:
    try:
        client = OpenAI()
    except:
        print("WARNING: OpenAI client initialization failed. Falling back to offline mode.")
        OFFLINE_MODE = True

# Define tools for OpenAI API
tools = [
    {
        "type": "function",
        "function": {
            "name": "locate_person",
            "description": "Command the drone to locate a person based on visual description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "point_of_interest": {
                        "type": "string",
                        "description": "Point of interest for the drone to focus on. (Like a yellow hat)"
                    },
                },
                "required": ["point_of_interest"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_distance",
            "description": "Adjust drone's distance to the object (move closer or away).",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["closer", "away"],
                        "description": "Direction to move relative to the object."
                    },
                },
                "required": ["direction"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spin_around",
            "description": "Make the drone perform a 360° spin around the current object of interest.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_drone_image",
            "description": "Get the current image from the drone camera.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    }
]

conversation_history = []

# Tello Controller Class
class TelloController():
    """Class to handle Tello drone control operations."""
    # Class variable to track if a stream is already active
    stream_active = False

    def __init__(self, connect_on_init=True):
        """Initialize the Tello controller.
        Args:
            connect_on_init (bool): Whether to connect to the drone upon initialization.
        """
        if TELLO_AVAILABLE:
            self.tello = Tello()
        else:
            self.tello = None  # Simulation mode

        self.is_flying = False
        self.frame_read = None
        self.mission_controller = DroneFaceTrackerOwl(tello=self.tello)

        if connect_on_init and TELLO_AVAILABLE:
            self.connect()

            # Only start video stream if not already active
            if not TelloController.stream_active:
                try:
                    print("Starting video stream for the first time")
                    self.tello.streamon()
                    TelloController.stream_active = True
                    time.sleep(1)  # Give time for stream to initialize
                    print("Stream started successfully")
                except Exception as e:
                    print(f"Error starting stream: {e}")
                    # If address already in use, stream is already active
                    if "address already in use" in str(e).lower():
                        TelloController.stream_active = True
                        print("Stream appears to be already active")
            else:
                print("Stream already active, not starting again")

    def connect(self):
        """Connect to the Tello drone."""
        if not TELLO_AVAILABLE:
            print("Simulated: Connected to Tello")
            return

        self.tello.connect()
        battery = self.tello.get_battery()
        print(f"Battery level: {battery}%")
        if battery < 20:
            print("WARNING: Battery level is low. Consider charging before mission.")

    def takeoff(self):
        """Take off the drone."""
        if not self.is_flying:
            if TELLO_AVAILABLE:
                self.tello.takeoff()
            self.is_flying = True
            if TELLO_AVAILABLE:
                self.move_up(70)

            print("Tello has taken off")
            # Allow some time to stabilize after takeoff
            time.sleep(1)
        else:
            print("Tello is already flying")

    def land(self):
        """Land the drone."""
        if self.is_flying:
            if TELLO_AVAILABLE:
                self.tello.land()
            self.is_flying = False
            print("Tello has landed")
        else:
            print("Tello is already on the ground")

    def move_up(self, distance_cm):
        """Move the drone up by the specified distance."""
        if self.is_flying:
            if TELLO_AVAILABLE:
                self.tello.move_up(distance_cm)
            print(f"Moved up {distance_cm}cm")
            time.sleep(1.5)  # Allow time to stabilize
        else:
            print("Cannot move: Tello is not flying")

    def move_closer(self, distance_cm=30):
        """Move the drone closer to the target."""
        if self.is_flying:
            if TELLO_AVAILABLE:
                self.tello.move_forward(distance_cm)
            print(f"Moved closer by {distance_cm}cm")
            time.sleep(1)
        else:
            print("Cannot move: Tello is not flying")

    def move_away(self, distance_cm=30):
        """Move the drone away from the target."""
        if self.is_flying:
            if TELLO_AVAILABLE:
                self.tello.move_back(distance_cm)
            print(f"Moved away by {distance_cm}cm")
            time.sleep(1)
        else:
            print("Cannot move: Tello is not flying")

    def rotate_360_by_steps(self, step_degrees=30):
        """Rotate the drone 360 degrees in specified step increments."""
        if not self.is_flying:
            print("Cannot rotate: Tello is not flying")
            return

        steps = 360 // step_degrees
        for i in range(steps):
            print(f"Rotating {step_degrees}° ({i+1}/{steps})")
            if TELLO_AVAILABLE:
                self.tello.rotate_clockwise(step_degrees)
            time.sleep(1)
        time.sleep(1)

    def get_current_frame(self):
        """Get the current frame from the drone camera."""
        if not TELLO_AVAILABLE:
            # Return a simulated frame in offline mode
            sample_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'sample_drone_view.jpg')
            if os.path.exists(sample_image_path):
                return cv2.imread(sample_image_path)
            # Create a simple dummy frame if no sample image exists
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "DRONE VIEW (SIMULATED)", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return dummy_frame

        # For real drone - simple direct approach for getting a frame
        try:
            # If stream is active but we don't have a frame reader, try to get one
            if TelloController.stream_active and (self.frame_read is None):
                try:
                    print("Trying to get frame reader for active stream")
                    self.frame_read = self.tello.get_frame_read().frame
                except Exception as e:
                    print(f"Error getting frame reader: {e}")
                    # If we can't get a frame reader, use sample image
                    return self._get_fallback_image()

            # If we have a frame reader, get the frame
            if self.frame_read is not None:
                try:
                    # Here's where we actually get the frame
                    frame = self.frame_read.frame
                    if frame is not None:
                        return frame
                except Exception as e:
                    print(f"Error getting frame from reader: {e}")

            # If we reach here, we couldn't get a valid frame
            return self._get_fallback_image()

        except Exception as e:
            print(f"Unexpected error in get_current_frame: {e}")
            return self._get_fallback_image()

    def _get_fallback_image(self):
        """Get a fallback image when frame capture fails."""
        # Try to use sample image first
        sample_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'sample_drone_view.jpg')
        if os.path.exists(sample_image_path):
            print("Using sample image fallback")
            return cv2.imread(sample_image_path)

        # Create a dummy frame if no sample image
        print("Creating dummy frame fallback")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "DRONE IMAGE UNAVAILABLE", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return dummy_frame

    def execute_floor_scan(self, target_texts, num_floors=3, floor_height_cm=40):
        """Execute a multi-floor scanning mission."""
        try:
            print(f"looking for: {target_texts}")
            self.mission_controller.run(connected=True, target_texts=target_texts)
            # self.takeoff()
            #
            # for floor in range(num_floors):
            #     print(f"\n--- Scanning Floor {floor + 1} ---")
            #     if floor > 0:
            #         self.move_up(floor_height_cm)
            #     self.rotate_360_by_steps(30)
            #     time.sleep(1)
            #
            # print("\n--- Mission Complete ---")
            # self.land()
        except Exception as e:
            print(f"Error during mission: {e}")
            self.land()

    def close(self):
        """Clean up resources."""
        if self.is_flying:
            self.land()
        if TELLO_AVAILABLE and self.tello:
            # No need to explicitly end the stream as this can be handled by the Tello SDK
            pass


# Global drone controller instance - TRUE SINGLETON
drone_controller = None

def get_drone_controller():
    """Get or initialize the drone controller using singleton pattern."""
    global drone_controller
    if drone_controller is None:
        print("Initializing drone controller for the first time")
        drone_controller = TelloController(connect_on_init=True)
    return drone_controller

def save_frame_as_image(frame, filename="current_frame.jpg"):
    """Save a frame as an image file."""
    if frame is None:
        return None

    # Ensure the static directory exists
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Save the image
    image_path = os.path.join(static_dir, filename)
    success = cv2.imwrite(image_path, frame)
    if success:
        print(f"Image saved successfully: {image_path}")
    else:
        print(f"Failed to save image: {image_path}")

    return filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global drone_controller
    user_message = request.json.get('message', '')
    action_results = []

    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_message})

    # Make sure drone controller is initialized
    try:
        controller = get_drone_controller()
    except Exception as e:
        print(f"Error initializing drone controller: {e}")
        return jsonify({
            "assistant_message": f"Error initializing drone: {str(e)}",
            "action_results": [],
            "offline_mode": OFFLINE_MODE,
            "error": True
        })

    if OFFLINE_MODE:
        # Offline mode: directly parse commands and execute functions
        function_calls = parse_offline_command(user_message)
        for function_call in function_calls:
            function_name = function_call["name"]
            arguments = function_call.get("arguments", {})
            try:
                result = execute_function(controller, function_name, arguments)
                action_results.append(result)
            except Exception as e:
                print(f"Error executing {function_name}: {e}")
                action_results.append({
                    "action": function_name,
                    "message": f"Error: {str(e)}",
                    "error": True
                })

        # Generate a simple response
        assistant_message = "Command processed in offline mode."
        conversation_history.append({"role": "assistant", "content": assistant_message})
    else:
        # Online mode: use OpenAI API
        # [rest of the online mode code remains unchanged]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            tools=tools,
            tool_choice="auto"
        )

        # Get the assistant's message
        assistant_message = response.choices[0].message

        # Add the assistant message to conversation history
        conversation_history.append(assistant_message.model_dump())

        # Process tool calls if any
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_id = tool_call.id
                arguments = json.loads(tool_call.function.arguments)

                # Execute the function call
                result = execute_function(controller, function_name, arguments)
                action_results.append(result)

                # Add tool response to conversation history
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": function_id,
                    "content": json.dumps(result)
                })

            # Get a follow-up response after tool use
            follow_up_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
                tools=tools,
                tool_choice="auto"
            )

            # Update assistant message with the follow-up response
            assistant_message = follow_up_response.choices[0].message
            conversation_history.append(assistant_message.model_dump())

            # Return for tool-based interactions
            return jsonify({
                "assistant_message": assistant_message.content,
                "action_results": action_results,
                "offline_mode": OFFLINE_MODE
            })

        # For non-tool messages
        assistant_message = assistant_message.content

    return jsonify({
        "assistant_message": assistant_message,
        "action_results": action_results,
        "offline_mode": OFFLINE_MODE
    })

def parse_offline_command(message):
    """Parse user message for drone commands in offline mode."""
    message = message.lower()
    function_calls = []

    # Look for person/object
    if "look for" in message or "find" in message or "search for" in message:
        # Extract what to look for
        target = message.split("for")[-1].strip() if "for" in message else "person"
        function_calls.append({
            "name": "locate_person",
            "arguments": {"point_of_interest": target}
        })

    # Move closer
    elif "closer" in message or "move in" in message or "approach" in message:
        function_calls.append({
            "name": "adjust_distance",
            "arguments": {"direction": "closer"}
        })

    # Move away
    elif "away" in message or "back up" in message or "retreat" in message:
        function_calls.append({
            "name": "adjust_distance",
            "arguments": {"direction": "away"}
        })

    # Spin around
    elif "spin" in message or "rotate" in message or "look around" in message or "turn" in message:
        function_calls.append({
            "name": "spin_around",
            "arguments": {}
        })

    # Get image
    elif "image" in message or "picture" in message or "photo" in message or "show me" in message or "camera" in message:
        function_calls.append({
            "name": "get_drone_image",
            "arguments": {}
        })

    # If no specific command is recognized, default to getting an image
    if not function_calls:
        function_calls.append({
            "name": "get_drone_image",
            "arguments": {}
        })

    return function_calls

def execute_function(controller, function_name, arguments):
    """Execute drone functions and return results."""
    try:
        # controller = get_drone_controller()

        if function_name == "locate_person":
            poi = arguments.get('point_of_interest', 'unknown')

            try:
                # Start mission thread
                mission_thread = threading.Thread(
                    target=controller.execute_floor_scan,
                    kwargs={"num_floors": 2, "floor_height_cm": 70, "target_texts": [f"a {poi}"]}
                )
                mission_thread.daemon = True
                mission_thread.start()

                return {
                    "action": "locate_person",
                    "message": f"Drone is scanning for {poi}."
                }
            except Exception as e:
                print(f"Error starting locate mission: {e}")
                return {
                    "action": "locate_person",
                    "message": f"Error starting mission: {e}",
                    "error": True
                }

        elif function_name == "adjust_distance":
            direction = arguments.get('direction', 'closer')
            if direction == "closer":
                controller.move_closer()
            else:
                controller.move_away()

            return {
                "action": "adjust_distance",
                "message": f"Drone is moving {direction}."
            }

        elif function_name == "spin_around":
            controller.rotate_360_by_steps(30)
            return {
                "action": "spin_around",
                "message": "Drone is performing a 360° spin around the object."
            }

        elif function_name == "get_drone_image":
            # Get the current frame
            frame = controller.get_current_frame()

            # Save the frame as an image
            image_filename = save_frame_as_image(frame)

            if image_filename:
                return {
                    "action": "get_drone_image",
                    "message": "Drone image captured successfully.",
                    "image": image_filename
                }
            else:
                return {
                    "action": "get_drone_image",
                    "message": "Failed to capture drone image.",
                    "error": True
                }

        else:
            return {
                "action": "unknown",
                "message": f"Unknown function: {function_name}",
                "error": True
            }
    except Exception as e:
        print(f"Unexpected error in execute_function: {e}")
        return {
            "action": function_name if function_name else "unknown",
            "message": f"Error executing function: {e}",
            "error": True
        }

@app.route('/drone_image/<filename>')
def drone_image(filename):
    """Serve drone images from the static directory."""
    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', filename)
    if os.path.exists(static_path):
        return send_file(static_path, mimetype='image/jpeg')
    else:
        return jsonify({"error": "Image not found"}), 404

@app.route('/toggle_mode')
def toggle_mode():
    """Toggle between online and offline mode."""
    global OFFLINE_MODE
    OFFLINE_MODE = not OFFLINE_MODE
    return jsonify({"offline_mode": OFFLINE_MODE})

@app.route('/status')
def status():
    """Get current drone and app status."""
    controller = get_drone_controller()
    return jsonify({
        "offline_mode": OFFLINE_MODE,
        "drone_connected": TELLO_AVAILABLE,
        "is_flying": controller.is_flying
    })

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the drone and release resources."""
    if drone_controller:
        drone_controller.close()
    return jsonify({"message": "Drone shutdown completed"})

if __name__ == '__main__':
    # Make sure resources are cleaned up on exit
    try:
        app.run(debug=True)
    finally:
        if drone_controller:
            drone_controller.close()
