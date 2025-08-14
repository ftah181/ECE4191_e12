import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import socket
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

# -------------------------
# Simulated Model Prediction
# -------------------------
def model_predict(frame):
    """
    Simulate a model that outputs bounding boxes on the frame.
    Replace this with your actual model inference code.
    """
    height, width, _ = frame.shape
    # Example: Draw a random bounding box
    x1, y1 = np.random.randint(0, width//2), np.random.randint(0, height//2)
    x2, y2 = x1 + np.random.randint(50, 150), y1 + np.random.randint(50, 150)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Class A", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

# -------------------------
# Simulated Voltage
# -------------------------
def get_voltage():
    """Simulate reading a voltage value (replace with ADC reading)."""
    return random.uniform(0, 5)


# -------------------------
# Tkinter GUI
# -------------------------
class UDPVideoApp:
    def __init__(self, root, udp_ip="0.0.0.0", udp_port=5005):
        self.root = root
        self.root.title("Model Output Display")

        # Setup UDP Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((udp_ip, udp_port))
        self.sock.setblocking(False)

        # Video capture
        # self.cap = cv2.VideoCapture(0) # Used while testing. Displays webcam
        self.video_label = tk.Label(root)
        self.video_label.pack(side=tk.LEFT)
        
        # --- Voltage Graph ---
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_ylim(0, 5)
        self.ax.set_title("Voltage vs Time")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Voltage (V)")
        self.x_data = []
        self.y_data = []
        self.line, = self.ax.plot([], [], 'r-')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.update()  # start loop
    
    def update(self):
        
        # Receive UDP Frame
        try:
            data, addr = self.sock.recvfrom(65536)
            npdata = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame = model_predict(frame)  # insert model prediction here
                
                # Convert to make compatible with Tkinter
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        except BlockingIOError:
            pass  # no frame received

        # Update Voltage Plot
        voltage = get_voltage()
        if len(self.x_data) >= 100:
            self.x_data = self.x_data[1:]
            self.y_data = self.y_data[1:]
        self.x_data.append(len(self.x_data))
        self.y_data.append(voltage)
        self.line.set_data(self.x_data, self.y_data)
        self.ax.set_xlim(0, max(100, len(self.x_data)))
        self.canvas.draw()

        # Schedule next update
        self.root.after(30, self.update)  # ~30 FPS

# -------------------------
# Run
# -------------------------
root = tk.Tk()
app = UDPVideoApp(root)
root.mainloop()
