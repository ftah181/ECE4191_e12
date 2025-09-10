import cv2
import socket
import numpy as np
import json
import time
import random

# -------------------------
# UDP Target Settings
# -------------------------
UDP_IP = "118.138.125.2"  # Change to your laptop's IP if on a network
UDP_PORT_VIDEO = 5005     # Port for video frames
UDP_PORT_ADC = 5006       # Port for ADC data

sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_adc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------------
# ADC Simulation Settings
# -------------------------
def simulate_adc_reading():
    """Simulate ADC reading from a single channel"""
    # Simulate voltage reading with some variation (0-3.3V range)
    voltage = round(1.65 + 0.8 * np.sin(time.time() * 0.5) + random.uniform(-0.1, 0.1), 3)
    voltage = max(0.0, min(3.3, voltage))  # Clamp between 0 and 3.3V
    
    reading = {
        'timestamp': time.time(),
        'voltage': voltage
    }
    return reading

# -------------------------
# Webcam Capture
# -------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

# Set lower resolution for Raspberry Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Timing variables
last_adc_time = 0
adc_interval = 0.1  # Send ADC data every 100ms (10Hz)

print(f"Sending video to {UDP_IP}:{UDP_PORT_VIDEO}")
print(f"Sending ADC data to {UDP_IP}:{UDP_PORT_ADC}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # --- Encode and send video frame ---
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ret:
            sock_video.sendto(buffer.tobytes(), (UDP_IP, UDP_PORT_VIDEO))

        # --- Send ADC data at specified interval ---
        current_time = time.time()
        if current_time - last_adc_time >= adc_interval:
            adc_data = simulate_adc_reading()
            adc_json = json.dumps(adc_data).encode('utf-8')
            sock_adc.sendto(adc_json, (UDP_IP, UDP_PORT_ADC))
            last_adc_time = current_time
            
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
    cap.release()
    cv2.destroyAllWindows()
    sock_video.close()
    sock_adc.close()