import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import socket
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import os
import threading
import queue
import time
from inference_sdk import InferenceHTTPClient

# --- Setup Roboflow client ---
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="lnHqcMh4NynT1If5FC38"  # Replace with your actual key
)

# Global variables for performance optimization
frame_skip_counter = 0
INFERENCE_SKIP_FRAMES = 20  # Run inference every N frames
last_predictions = []  # Cache last predictions

# -------------------------
# Simulated Model Prediction
# -------------------------
def model_predict_sim(frame):
    """
    Simulate a model that outputs bounding boxes on the frame.
    """
    height, width, _ = frame.shape
    # Example: Draw a random bounding box
    x1, y1 = np.random.randint(0, width//2), np.random.randint(0, height//2)
    x2, y2 = x1 + np.random.randint(50, 150), y1 + np.random.randint(50, 150)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Class A", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame, [{"class": "Class A", "confidence": 0.85, "x": (x1+x2)//2, "y": (y1+y2)//2, "width": x2-x1, "height": y2-y1}]

# -------------------------
# Optimized Inference Model
# -------------------------
def model_predict_fast(frame, force_inference=False):
    """
    Optimized inference with frame skipping and caching
    """
    global frame_skip_counter, last_predictions
    
    # Skip inference on some frames for speed
    frame_skip_counter += 1
    if not force_inference and frame_skip_counter < INFERENCE_SKIP_FRAMES:
        # Use cached predictions and just draw them
        return draw_predictions(frame, last_predictions)
    
    frame_skip_counter = 0
    
    # Resize frame for faster inference (smaller = faster)
    original_shape = frame.shape[:2]
    inference_size = (320, 240)  # Smaller size for faster inference
    resized_frame = cv2.resize(frame, inference_size)
    
    predictions = []
    
    try:
        # Method 1: Try with PIL Image (most compatible)
        from PIL import Image
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Run inference with PIL Image
        result = client.infer(pil_image, model_id="animal-detection-evlon/1")
        
        # Scale predictions back to original size
        scale_x = original_shape[1] / inference_size[0]
        scale_y = original_shape[0] / inference_size[1]
        
        predictions = []
        for pred in result.get("predictions", []):
            scaled_pred = pred.copy()
            scaled_pred["x"] = int(pred["x"] * scale_x)
            scaled_pred["y"] = int(pred["y"] * scale_y)
            scaled_pred["width"] = int(pred["width"] * scale_x)
            scaled_pred["height"] = int(pred["height"] * scale_y)
            predictions.append(scaled_pred)
        
        last_predictions = predictions  # Cache for next frames
        print(f"Fast inference: {len(predictions)} detections")
        
    except Exception as inference_error:
        print(f"Inference failed: {inference_error}")
        # Use simulation for speed if inference fails
        return model_predict_sim(frame)
    
    return draw_predictions(frame, predictions)

def draw_predictions(frame, predictions):
    """
    Separate function to draw predictions on frame
    """
    try:
        for pred in predictions:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            class_name = pred["class"]
            confidence = pred["confidence"]

            # Convert center coordinates to corner coordinates
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
    
    except Exception as draw_error:
        print(f"Error drawing predictions: {draw_error}")
    
    return frame, predictions

# -------------------------
# Threaded Inference Worker
# -------------------------
class InferenceWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to prevent lag
        self.result_queue = queue.Queue(maxsize=5)
        self.running = True
        
    def add_frame(self, frame):
        try:
            # Clear old frames if queue is full
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Skip frame if queue is full
    
    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                processed_frame, predictions = model_predict_fast(frame, force_inference=True)
                
                # Clear old results
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        break
                
                self.result_queue.put_nowait((processed_frame, predictions))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference worker error: {e}")
    
    def stop(self):
        self.running = False

# -------------------------
# Simulated Voltage (optimized)
# -------------------------
def get_voltage():
    """Simulate reading a voltage value."""
    return random.uniform(0, 5)

# -------------------------
# Optimized Tkinter GUI
# -------------------------
class UDPVideoApp:
    def __init__(self, root, udp_ip="0.0.0.0", udp_port=5005):
        self.root = root
        self.root.title("Model Output Display - Optimized")
        self.root.geometry("1400x700")  # Larger window for bigger video display

        # Performance settings
        self.use_threading = tk.BooleanVar(value=True)
        self.skip_inference = tk.BooleanVar(value=False)
        
        # Setup UDP Socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((udp_ip, udp_port))
            self.sock.setblocking(False)
            print(f"UDP socket bound to {udp_ip}:{udp_port}")
        except Exception as e:
            print(f"Socket binding error: {e}")

        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.display_fps = 0
        
        # Threading for inference
        self.inference_worker = InferenceWorker()
        self.inference_worker.start()
        self.last_inference_result = None
        
        # Frame for video and controls
        self.video_frame = tk.Frame(root)
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Performance controls
        self.controls_frame = tk.Frame(self.video_frame)
        self.controls_frame.pack()
        
        tk.Checkbutton(self.controls_frame, text="Threaded Inference", 
                      variable=self.use_threading).pack(side=tk.LEFT)
        tk.Checkbutton(self.controls_frame, text="Skip Inference (Display Only)", 
                      variable=self.skip_inference).pack(side=tk.LEFT)
        
        # Status labels
        self.status_label = tk.Label(self.video_frame, text="Waiting for UDP frames...", 
                                   font=("Arial", 10))
        self.status_label.pack()
        
        self.fps_label = tk.Label(self.video_frame, text="FPS: 0", 
                                font=("Arial", 10), fg="blue")
        self.fps_label.pack()

        # Video display with proper sizing
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack()
        
        # Frame counter
        self.frame_count = 0
        self.frame_counter_label = tk.Label(self.video_frame, text="Frames: 0")
        self.frame_counter_label.pack()
        
        # Testing mode toggle
        self.use_webcam = tk.BooleanVar()
        self.webcam_checkbox = tk.Checkbutton(self.video_frame, text="Use Webcam for Testing", 
                                            variable=self.use_webcam)
        self.webcam_checkbox.pack()
        
        # Initialize webcam capture (for testing)
        self.cap = None
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                # Set webcam to lower resolution for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                print("Webcam available for testing")
            else:
                self.cap = None
        except:
            print("Could not initialize webcam")
            self.cap = None

        # --- Optimized Voltage Graph ---
        self.fig, self.ax = plt.subplots(figsize=(4, 3))  # Smaller plot
        self.ax.set_ylim(0, 5)
        self.ax.set_title("Voltage vs Time")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Voltage (V)")
        self.x_data = []
        self.y_data = []
        self.line, = self.ax.plot([], [], 'r-')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Voltage update counter (update less frequently)
        self.voltage_update_counter = 0

        self.update()  # start loop
    
    def calculate_fps(self):
        """Calculate and update FPS display"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.display_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            self.fps_label.config(text=f"FPS: {self.display_fps:.1f}")
    
    def update(self):
        frame = None
        
        # Try to get frame from UDP first
        try:
            data, addr = self.sock.recvfrom(65536)
            npdata = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.status_label.config(text=f"UDP: {addr[0]}:{addr[1]}")
                self.frame_count += 1
        except socket.error:
            pass
        
        # Fallback to webcam if enabled and no UDP
        if frame is None and self.use_webcam.get() and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.status_label.config(text="Using webcam")
                self.frame_count += 1
        
        # Process frame if available
        if frame is not None:
            try:
                # Skip inference if requested (display only mode)
                if self.skip_inference.get():
                    processed_frame = frame
                elif self.use_threading.get():
                    # Threaded inference
                    self.inference_worker.add_frame(frame)
                    
                    # Check for completed inference
                    result = self.inference_worker.get_result()
                    if result is not None:
                        self.last_inference_result = result
                    
                    # Use last result or original frame
                    if self.last_inference_result is not None:
                        processed_frame, _ = self.last_inference_result
                    else:
                        processed_frame = frame
                else:
                    # Direct inference (slower)
                    processed_frame, _ = model_predict_fast(frame)
                
                # Resize for display if too large
                display_frame = self.resize_for_display(processed_frame)
                
                # Convert and display
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                # Update counters
                self.frame_counter_label.config(text=f"Frames: {self.frame_count}")
                self.calculate_fps()
                
            except Exception as e:
                print(f"Frame processing error: {e}")

        # Update voltage plot less frequently for better performance
        self.voltage_update_counter += 1
        if self.voltage_update_counter >= 3:  # Update every 3 frames
            self.voltage_update_counter = 0
            try:
                voltage = get_voltage()
                if len(self.x_data) >= 50:  # Smaller buffer
                    self.x_data = self.x_data[1:]
                    self.y_data = self.y_data[1:]
                self.x_data.append(len(self.x_data))
                self.y_data.append(voltage)
                self.line.set_data(self.x_data, self.y_data)
                self.ax.set_xlim(0, max(50, len(self.x_data)))
                self.canvas.draw()
            except Exception as e:
                print(f"Plot update error: {e}")

        # Faster update cycle
        self.root.after(16, self.update)  # ~60 FPS target
    
    def resize_for_display(self, frame):
        """Resize frame for display with reasonable size limits"""
        height, width = frame.shape[:2]
        max_width, max_height = 800, 600  # Much larger display size
        min_width, min_height = 320, 240  # Minimum size to prevent tiny display
        
        # Only resize if frame is too large OR too small
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        elif width < min_width or height < min_height:
            scale = max(min_width/width, min_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        
        return frame

    def __del__(self):
        if hasattr(self, 'inference_worker'):
            self.inference_worker.stop()
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'sock'):
            self.sock.close()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = UDPVideoApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
    finally:
        if hasattr(app, 'inference_worker'):
            app.inference_worker.stop()
        if hasattr(app, 'cap') and app.cap is not None:
            app.cap.release()
        if hasattr(app, 'sock'):
            app.sock.close()