import threading
import socket
import queue
import numpy as np
import cv2
import time

class VideoReceiver(threading.Thread):
    def __init__(self, udp_ip, video_port, max_queue=1):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576 * 4)
        self.sock.bind((udp_ip, video_port))
        self.sock.setblocking(False)
        self.frame_queue = queue.Queue(maxsize=max_queue)
        self.running = True

    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1048576)
                npdata = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
                if frame is not None:
                    # Always keep only the latest frame
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.frame_queue.put_nowait(frame)
            except socket.error:
                time.sleep(0.001)
            except Exception:
                continue

    def get_latest_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.sock.close()