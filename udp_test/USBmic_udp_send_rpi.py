import cv2
import socket
import numpy as np
import json
import time
from picamera2 import Picamera2
import pyaudio

# -------------------------
# UDP Target Settings
# -------------------------
UDP_IP = "172.20.10.2"
UDP_PORT_VIDEO = 5005   # Port for video frames
UDP_PORT_AUDIO = 5006   # Port for audio JSON

CHUNK = 1024            # Number of audio samples per chunk
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------------
# Audio Setup
# -------------------------
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# -------------------------
# Camera Setup
# -------------------------
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (320, 240)},
)
picam2.configure(config)
picam2.start()

print(f"Sending video to {UDP_IP}:{UDP_PORT_VIDEO}")
print(f"Sending audio JSON to {UDP_IP}:{UDP_PORT_AUDIO}")

try:
    while True:
        # --- Capture and send video ---
        frame = picam2.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        if ret:
            sock_video.sendto(buffer.tobytes(), (UDP_IP, UDP_PORT_VIDEO))

        # --- Capture and send audio as JSON ---
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(audio_data, dtype=np.int16).tolist()

        audio_json = json.dumps({"voltages": samples}).encode("utf-8")
        sock_audio.sendto(audio_json, (UDP_IP, UDP_PORT_AUDIO))

        # Debug print
        print(f"Sent {len(samples)} samples")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    sock_video.close()
    sock_audio.close()
    stream.stop_stream()
    stream.close()
    p.terminate()
