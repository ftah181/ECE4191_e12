import cv2
import socket
import numpy as np
import json
import time
import threading
from picamera2 import Picamera2
import spidev
import RPi.GPIO as GPIO


#----UDP Target Settings----
UDP_IP = "172.20.10.2"
UDP_PORT_VIDEO=5005
UDP_PORT_ADC =5006

sock_video = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock_adc = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)


#----ADC Setup----
CS_PIN = 21
GPIO.setmode(GPIO.BCM)
GPIO.setup(CS_PIN, GPIO.OUT)
GPIO.output(CS_PIN, GPIO.HIGH)
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 3600000

def read_mcp3008(channel):
    if channel < 0 or channel > 7:
        return -1
    cmd = [1, (8 + channel) << 4, 0]
    GPIO.output(CS_PIN, GPIO.LOW)
    result = spi.xfer2(cmd)
    GPIO.output(CS_PIN, GPIO.HIGH)
    value = ((result[1] & 3) << 8) + result[2]
    return value

# Camera Setup
# -------------------------
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

#Threads
def video_thread():
    """Thread for capturing and sending video frames"""
    print(f"Sending video to {UDP_IP}:{UDP_PORT_VIDEO}")
    while running:
        frame = picam2.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        if ret:
            sock_video.sendto(buffer.tobytes(), (UDP_IP, UDP_PORT_VIDEO))

        time.sleep(0.01)  # Prevent network flooding


def adc_thread():
    """Thread for reading ADC and sending samples"""
    print(f"Sending ADC data to {UDP_IP}:{UDP_PORT_ADC}")
    
    #----For testing----
    SAMPLE_WINDOW = 100 #
    sample_count = 0
    start_time = time.time()

    while running:
        value = read_mcp3008(0)
        adc_data = {"voltage": value}
        adc_json = json.dumps(adc_data).encode("utf-8")
        sock_adc.sendto(adc_json, (UDP_IP, UDP_PORT_ADC))

        # Debug print
        #print(f"ADC: {adc_data}")

        #----For testing----
        sample_count += 1
        if sample_count >= SAMPLE_WINDOW:
            end_time = time.time()
            elapsed = end_time - start_time
            actual_rate = sample_count / elapsed
            print(f"Achieved sampling rate: {actual_rate:.1f} Hz over {elapsed:.2f} seconds")
            sample_count = 0
            start_time = time.time()

# -------------------------
# Main
# -------------------------
running = True
t1 = threading.Thread(target=video_thread, daemon=True)
t2 = threading.Thread(target=adc_thread, daemon=True)

try:
    t1.start()
    t2.start()

    while True:  # Keep main thread alive
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping...")
    running = False

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    sock_video.close()
    sock_adc.close()
    GPIO.cleanup()