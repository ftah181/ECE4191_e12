import cv2
import socket
import numpy as np
import json
import time
import random
from picamera2 import Picamera2
import spidev
import RPi.GPIO as GPIO
import time
# -------------------------
# UDP Target Settings
# -------------------------
UDP_IP = "192.168.99.175"  # Change to your laptop's IP if on a network
UDP_PORT_VIDEO = 5005     # Port for video frames
UDP_PORT_ADC = 5006       # Port for ADC data

sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_adc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------------
# ADC Simulation Settings
# -------------------------
# Setup
CS_PIN = 8  # GPIO21 as CS
GPIO.setmode(GPIO.BCM)
GPIO.setup(CS_PIN, GPIO.OUT)
GPIO.output(CS_PIN, GPIO.HIGH)  # CS idle high
spi = spidev.SpiDev()
spi.open(0, 0)  # SPI bus 0, device 0
spi.max_speed_hz = 1350000  # 1.35 MHz
def read_mcp3008(channel):
    if channel < 0 or channel > 7:
        return -1
    cmd = [1, (8 + channel) << 4, 0]
    
    GPIO.output(CS_PIN, GPIO.LOW)     # Select MCP3008
    result = spi.xfer2(cmd)
    GPIO.output(CS_PIN, GPIO.HIGH)    # Deselect MCP3008

    value = ((result[1] & 3) << 8) + result[2]  # 10-bit value
    return value

# -------------------------
# Webcam Capture
# -------------------------

# Configure camera
picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={
    "size": (640, 480),
    })

picam2.configure(config)

# controls = {
#     "AwbMode": 4,
#     "AeEnable": True,                  # enable auto exposure
#     #"ColourGains": (1.2, 1.1),  # boost red slightly, blue slightly
#     "AnalogueGain": 1.0                # default gain
# }
#picam2.set_controls(controls)

picam2.start()

# Timing variables
last_adc_time = 0
adc_interval = 0.1  # Send ADC data every 100ms (10Hz)

print(f"Sending video to {UDP_IP}:{UDP_PORT_VIDEO}")
print(f"Sending ADC data to {UDP_IP}:{UDP_PORT_ADC}")

try:
    while True:
        frame = picam2.capture_array()

        # --- Encode and send video frame ---
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        if ret:
            sock_video.sendto(buffer.tobytes(), (UDP_IP, UDP_PORT_VIDEO))

        # --- Send ADC data at specified interval ---
        start_time = time.time()
        adc_data = read_mcp3008(0)
        adc_json = json.dumps(adc_data).encode('utf-8')
        sock_adc.sendto(adc_json, (UDP_IP, UDP_PORT_ADC))
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        actual_rate = 1 / elapsed
        print(f"Achieved sampling rate: {actual_rate:.1f} Hz over {elapsed:.2f} seconds")
            
            # Optional: print ADC values for debugging
            # print(f"ADC: {adc_data['voltage']}V")

        # Optional: show local preview
        # cv2.imshow('Local Webcam', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Small delay to prevent overwhelming the network
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    sock_video.close()
    sock_adc.close()