import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
import numpy as np
import cv2
from pynput import keyboard
from utils import *
from pydnet import *
from tracker import Tracker
from djitellopy import Tello

parser = argparse.ArgumentParser(description='Argument parser')

parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--resolution', dest='resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/IROS18/pydnet', help='checkpoint directory')

args = parser.parse_args()

class TelloCV:
    def __init__(self):
        self.drone = Tello()
        self.drone.connect()
        self.drone.streamon()
        self.frame_reader = self.drone.get_frame_read()
        self.speed = 50
        self.keydown = False
        self.init_controls()

    def init_controls(self):
        self.controls = {
            'w': 'move_forward',
            's': 'move_back',
            'a': 'move_left',
            'd': 'move_right',
            'Key.space': 'move_up',
            'Key.shift': 'move_down',
            'q': 'rotate_counter_clockwise',
            'e': 'rotate_clockwise',
            'Key.tab': lambda: self.drone.takeoff(),
            'Key.backspace': lambda: self.drone.land(),
        }
        self.key_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.key_listener.start()

    def on_press(self, key):
        if self.keydown:
            return
        try:
            self.keydown = True
            keyname = str(key).strip('"')
            print('+' + keyname)
            if keyname == 'Key.esc':
                self.drone.streamoff()
                self.drone.end()
                exit(0)
            if keyname in self.controls:
                action = self.controls[keyname]
                if isinstance(action, str):
                    getattr(self.drone, action)(self.speed)
                else:
                    action()
        except Exception as e:
            print(f"Error in on_press: {e}")

    def on_release(self, key):
        self.keydown = False
        keyname = str(key).strip('"')
        print('-' + keyname)
        # Could implement stop logic here if needed

    def process_frame(self):
        return self.frame_reader.frame

def main(_):
    tellotrack = TelloCV()

    with tf.Graph().as_default():
        height = args.height
        width = args.width
        placeholders = {'im0': tf.placeholder(tf.float32, [None, None, None, 3], name='im0')}

        with tf.variable_scope("model") as scope:
            model = pydnet(placeholders)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        loader = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            loader.restore(sess, args.checkpoint_dir)

            while True:
                frame = tellotrack.process_frame()
                img = cv2.resize(frame, (width, height)).astype(np.float32) / 255.
                img = np.expand_dims(img, 0)
                start = time.time()
                disp = sess.run(model.results[args.resolution - 1], feed_dict={placeholders['im0']: img})
                end = time.time()

                disp_color = applyColorMap(disp[0, :, :, 0] * 20, 'plasma')
                toShow = (np.concatenate((img[0], disp_color), 0) * 255.).astype(np.uint8)
                toShow = cv2.resize(toShow, (int(width / 2), height))

                cv2.imshow('pydnet', toShow)
                k = cv2.waitKey(1)
                if k == 27:  # ESC key
                    break
                if k == ord('p'):
                    cv2.waitKey(0)

                print("Time: " + str(end - start))

if __name__ == '__main__':
    tf.app.run()
