import time
from djitellopy import Tello

### Hardcoded 360 degree fly around an object  #####

class MotionController:
    def __init__(self, config=None):
        self.tello = Tello()
        self._running = False
        
        # Default motion parameters
        self.config = {
            'takeoff_height': 100,      # cm
            'ascent_speed': 20,         # 0-100
            'forward_speed': 15,        # 0-100
            'sideward_speed': 0,        # 0-100 (positive = right, negative = left)
            'yaw_speed': 15,            # 0-100
            'motion_angle': 90,         # degrees
            'hover_time': 1.0,          # seconds
            'min_battery': 20           # percent
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)

    def set_motion_params(self, **kwargs):
        """Update motion parameters"""
        self.config.update(kwargs)
        print("Updated motion parameters:", self.config)

    def start(self):
        """Start the motion sequence"""
        try:
            # Connect to Tello
            print("Connecting to Tello...")
            self.tello.connect()
            
            # Check battery
            battery = self.tello.get_battery()
            print(f"Battery: {battery}%")
            if battery < self.config['min_battery']:
                print(f"WARNING: Low battery ({battery}%). Please charge.")
                return

            # Takeoff
            print("Taking off...")
            self.tello.takeoff()
            time.sleep(2)

            # Ascend to specified height
            print(f"Ascending to {self.config['takeoff_height']}cm...")
            while self.tello.get_height() < self.config['takeoff_height']:
                self.tello.send_rc_control(0, 0, self.config['ascent_speed'], 0)
                time.sleep(0.1)
            self.tello.send_rc_control(0, 0, 0, 0)
            time.sleep(self.config['hover_time'])

            # Execute motion
            print(f"Executing {self.config['motion_angle']}° motion...")
            arc_time = self.config['motion_angle'] / self.config['yaw_speed']
            
            # Combine forward and sideward motion with rotation
            self.tello.send_rc_control(
                self.config['sideward_speed'],  # left/right
                self.config['forward_speed'],   # forward/backward
                0,                              # up/down
                self.config['yaw_speed']        # rotation
            )
            time.sleep(arc_time)
            self.tello.send_rc_control(0, 0, 0, 0)
            time.sleep(self.config['hover_time'])

            # Land
            print("Landing...")
            self.tello.land()
            time.sleep(2)

        except Exception as e:
            print(f"Error in motion sequence: {str(e)}")
            # Ensure we land in case of error
            try:
                self.tello.land()
            except:
                pass

def main():
    """Test the motion sequence with custom parameters"""
    # Example custom configuration
    custom_config = {
        'takeoff_height': 100,       # Height in cm
        'ascent_speed': 25,         # Vertical speed
        'forward_speed': 30,        # Forward speed
        'sideward_speed': -30,       # Sideward speed (right)
        'yaw_speed': 50,            # Rotation speed
        'motion_angle': 450,        # Full circle
        'hover_time': 2.5,          # Hover time between motions
        'min_battery': 15           # Minimum battery threshold
    }
    
    controller = MotionController(custom_config)
    print("Starting motion sequence with custom parameters...")
    controller.start()

if __name__ == "__main__":
    main()