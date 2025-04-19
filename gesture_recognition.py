import serial
import numpy as np
import time
import pickle
import os

# Open the serial port
try:
    # List all available serial ports
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    print("Available ports:")
    for port in ports:
        print(f"- {port.device}")

    ser = serial.Serial('/dev/tty.usbmodem93106601', 38400)
    # print("Successfully opened serial port")
    # print(f"Port settings:")
    # print(f"- Baudrate: {ser.baudrate}")
    # print(f"- Bytesize: {ser.bytesize}")
    # print(f"- Parity: {ser.parity}")
    # print(f"- Stopbits: {ser.stopbits}")
    # print(f"- Timeout: {ser.timeout}")
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    raise

def detectSnap(gx1, gx2,valid):
    if (gx2-gx1)>50000 and valid:
        detect = 1
        valid = False
    elif (gx2-gx1)<50000 and not valid:
        detect = 0
        valid = True
    else:
        detect = 0
    return detect, valid


def write_pickle_atomic(data, file_path):
    temp_file_path = f"{file_path}.tmp"
    with open(temp_file_path, 'wb') as f:
        pickle.dump(data, f)
    os.replace(temp_file_path, file_path)  # Atomic rename

try:
    if ser.is_open:
        print("Serial port is open.")
    nextLine = None
    allSensor1 = []
    allSensor2 = []
    snapCount = 0
    allDetect = []
    valid = True
    detect = 0
    currentTime = None
    allSnaps = []
    write_pickle_atomic(allSnaps,'cache.pkl')
    while True:
        # Read a line of data from the serial port
        data = ser.readline().decode('utf-8').strip()
        # print(data)
        if data:

            dataVals = [int(x) for x in data.split(", ")]

            allSensor1.append([dataVals[0]])
            allSensor2.append([dataVals[1]])
            if len(allSensor1)>0 and len(allSensor2)>0:
                detect,valid = detectSnap(np.array(allSensor1).transpose()[0][-1],np.array(allSensor2).transpose()[0][-1], valid)
            if detect == 1:

                if snapCount ==0:
                    snapCount =1
                    prevTime = time.time()
                else:
                    currentTime = time.time()
                    if currentTime - prevTime < 0.7:
                        snapCount +=1
                        prevTime = time.time()
                        
                print("snap detected!")
            elif snapCount>0:
                currentTime = time.time()

                if currentTime - prevTime >0.7:
                    print("snaps: ", snapCount)
                    allSnaps.append(snapCount)
                    write_pickle_atomic(allSnaps,'cache.pkl')
                    print(allSnaps)
                    snapCount =0
                

except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
    print("Serial port closed.")