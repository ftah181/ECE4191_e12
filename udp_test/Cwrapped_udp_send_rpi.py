import cv2
import socket
import numpy as np
import json
import time
from picamera2 import Picamera2
import ctypes
import os

# -------------------------
# UDP Target Settings
# -------------------------
UDP_IP = "172.20.10.2"  # Change to your laptop's IP
UDP_PORT_VIDEO = 5005    # Port for video frames
UDP_PORT_ADC = 5006      # Port for ADC data

sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_adc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------------
# Load C library for MCP3008
# -------------------------
lib_path = os.path.abspath("libmcp3008.so")
lib = ctypes.CDLL(lib_path)

lib.sample_mcp3008.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint16)]
lib.sample_mcp3008.restype = ctypes.c_double

# Pre-allocate buffer for a single sample
_single_sample = (ctypes.c_uint16 * 1)()

def read_mcp3008_c(channel):
    """
    Reads a single value from MCP3008 using the C library.
    Returns the 10-bit ADC value (0-1023)
    """
    lib.sample_mcp3008(channel, 1, _single_sample)
    return _single_sample[0]

# -------------------------
# Configure Camera
# -------------------------
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

# -------------------------
# Sampling & Video Loop
# -------------------------
SAMPLE_WINDOW = 100  # Number of samples to average for rate
sample_count = 0
start_time = time.time()

print(f"Sending video to {UDP_IP}:{UDP_PORT_VIDEO}")
print(f"Sending ADC data to {UDP_IP}:{UDP_PORT_ADC}")

try:
    while True:
        # --- Capture video frame ---
        frame = picam2.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        if ret:
            sock_video.sendto(buffer.tobytes(), (UDP_IP, UDP_PORT_VIDEO))

        # --- Read ADC using C library ---
        adc_value = read_mcp3008_c(0)
        adc_data = {"voltage": adc_value}  # You can scale if needed
        adc_json = json.dumps(adc_data).encode('utf-8')
        sock_adc.sendto(adc_json, (UDP_IP, UDP_PORT_ADC))

        # --- Sampling rate calculation ---
        sample_count += 1
        if sample_count >= SAMPLE_WINDOW:
            end_time = time.time()
            elapsed = end_time - start_time
            actual_rate = sample_count / elapsed
            print(f"Achieved sampling rate: {actual_rate:.1f} Hz over {elapsed:.2f} seconds")
            sample_count = 0
            start_time = time.time()

        # Optional: print ADC values
        print(f"ADC: {adc_value}")

        # Optional: small delay to prevent network overload
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    sock_video.close()
    sock_adc.close()
