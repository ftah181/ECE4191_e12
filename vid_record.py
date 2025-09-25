import socket
import cv2
import numpy as np
import time
import os

# UDP setup
UDP_IP = "0.0.0.0"   # Listen on all interfaces
UDP_PORT = 5005
BUFFER_SIZE = 65535  # Max UDP packet size

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP video on {UDP_IP}:{UDP_PORT}...")

# Video writer setup (lazy init after first frame so we know dimensions)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = None
fps = 20.0  # Adjust to match sender's frame rate

while True:
    try:
        data, addr = sock.recvfrom(BUFFER_SIZE)

        # Decode JPEG to OpenCV image
        npdata = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)

        if frame is not None:
            # Initialize writer when we know frame size
            if out is None:
                height, width, _ = frame.shape
                out = cv2.VideoWriter("output3.mp4", fourcc, fps, (width, height))
                print(f"Started recording: {width}x{height} at {fps} FPS")

            out.write(frame)  # Append frame to .mp4
            print("Frame written")
        else:
            print("⚠️ Failed to decode frame")

    except KeyboardInterrupt:
        print("Stopped by user")
        break

# Cleanup
if out is not None:
    out.release()
sock.close()
