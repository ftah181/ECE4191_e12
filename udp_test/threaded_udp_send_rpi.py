import cv2
import socket
import numpy as np
import json
import time
import threading
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

NUM_SAMPLES_PER_CALL = 10
_buffer = (ctypes.c_uint16 * NUM_SAMPLES_PER_CALL)()

def read_mcp3008_c(channel, num_samples=NUM_SAMPLES_PER_CALL):
    """Reads multiple samples from MCP3008 using the C library."""
    lib.sample_mcp3008(channel, num_samples, _buffer)
    return list(_buffer[:num_samples])

# -------------------------
# Configure Camera
# -------------------------
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

# -------------------------
# Thread Functions
# -------------------------
def video_thread():
    print(f"[VIDEO] Sending video to {UDP_IP}:{UDP_PORT_VIDEO}")
    while running:
        frame = picam2.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        if ret:
            sock_video.sendto(buffer.tobytes(), (UDP_IP, UDP_PORT_VIDEO))
        time.sleep(0.01)  # prevent network overload

def adc_thread():
    print(f"[ADC] Sending ADC data to {UDP_IP}:{UDP_PORT_ADC}")
    SAMPLE_WINDOW = 100
    sample_count = 0
    start_time = time.time()

    while running:
        adc_values = read_mcp3008_c(0)
        adc_data = {"voltages": adc_values}
        adc_json = json.dumps(adc_data).encode('utf-8')
        sock_adc.sendto(adc_json, (UDP_IP, UDP_PORT_ADC))

        sample_count += 1
        if sample_count >= SAMPLE_WINDOW:
            elapsed = time.time() - start_time
            actual_rate = (sample_count * NUM_SAMPLES_PER_CALL) / elapsed
            print(f"[ADC] Achieved sampling rate: {actual_rate:.1f} Hz over {elapsed:.2f} s")
            sample_count = 0
            start_time = time.time()

        print(f"[ADC] {adc_values}")
        time.sleep(0.01)

# -------------------------
# Main
# -------------------------
running = True

t1 = threading.Thread(target=video_thread, daemon=True)
t2 = threading.Thread(target=adc_thread, daemon=True)

t1.start()
t2.start()

try:
    while True:
        time.sleep(1)  # Keep main thread alive
except KeyboardInterrupt:
    print("\nStopping...")
    running = False
    t1.join()
    t2.join()
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    sock_video.close()
    sock_adc.close()
