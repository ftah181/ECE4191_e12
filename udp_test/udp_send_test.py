import cv2
import socket
import numpy as np

# -------------------------
# UDP Target Settings
# -------------------------
UDP_IP = "192.168.1.119"  # Change to your laptop's IP if on a network
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------------
# Webcam Capture
# -------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

# Set lower resolution for Raspberry Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # --- Encode frame as JPEG ---
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ret:
        continue

    # --- Send over UDP ---
    sock.sendto(buffer.tobytes(), (UDP_IP, UDP_PORT))

    # Optional: show local preview
    # cv2.imshow('Local Webcam', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
