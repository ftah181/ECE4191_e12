import cv2
import socket
import numpy as np

# -------------------------
# UDP Target Settings
# -------------------------
UDP_IP = "118.138.103.246"  # Change to your laptop's IP if on a network
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------------
# Webcam Capture
# -------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # --- Encode frame as JPEG ---
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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
