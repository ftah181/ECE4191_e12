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
import json
from inference import get_model
from inference_sdk import InferenceHTTPClient
from roboflow import Roboflow
from ultralytics import YOLO

# --- Setup Roboflow client ---
# client = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key="lnHqcMh4NynT1If5FC38"  # Replace with your actual key
# )

model = get_model("animal-detection-evlon/1", api_key="lnHqcMh4NynT1If5FC38")

rf = Roboflow(api_key="lnHqcMh4NynT1If5FC38")
project = rf.workspace("4191").project("animal-detection-evlon")
version = project.version(3)
dataset = version.download("yolov8")
model_path = dataset.location + "/model/best.pt"
model = YOLO(model_path)

# Global variables for performance optimization
frame_skip_counter = 0
INFERENCE_SKIP_FRAMES = 20  # Run inference every N frames (increased from 3)
last_predictions = []  # Cache last predictions
frame_buffer = None  # Buffer for frame reuse

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
# Ultra-Fast Inference Model
# -------------------------
def model_predict_ultra_fast(frame, force_inference=False):
    """
    Ultra-optimized inference with aggressive optimizations
    """
    global frame_skip_counter, last_predictions, frame_buffer
    
    # Skip inference on most frames for maximum speed
    frame_skip_counter += 1
    if not force_inference and frame_skip_counter < INFERENCE_SKIP_FRAMES:
        # Use cached predictions and just draw them
        return draw_predictions_fast(frame, last_predictions)
    
    frame_skip_counter = 0
    
    # Even smaller inference size for maximum speed
    original_shape = frame.shape[:2]
    inference_size = (160, 120)  # Very small for ultra-fast inference
    
    # Use cached resized frame if available
    if frame_buffer is None or frame_buffer.shape[:2] != inference_size:
        resized_frame = cv2.resize(frame, inference_size)
        frame_buffer = resized_frame  # Cache for potential reuse
    else:
        resized_frame = cv2.resize(frame, inference_size)
    
    predictions = []
    
    try:
        # Fast PIL conversion without extra copies
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Run inference
        result = model(pil_image)
        print(result)
        
        # Quick scaling back to original size
        scale_x = original_shape[1] / inference_size[0]
        scale_y = original_shape[0] / inference_size[1]
        
        # Optimized prediction scaling
        predictions = []
        for pred in result.get("predictions", []):
            predictions.append({
                "class": pred["class"],
                "confidence": pred["confidence"],
                "x": int(pred["x"] * scale_x),
                "y": int(pred["y"] * scale_y),
                "width": int(pred["width"] * scale_x),
                "height": int(pred["height"] * scale_y)
            })
        
        last_predictions = predictions
        
    except Exception as inference_error:
        # Silent fallback to cached predictions
        predictions = last_predictions
    
    return draw_predictions_fast(frame, predictions)

def draw_predictions_fast(frame, predictions):
    """
    Optimized drawing with minimal operations
    """
    try:
        # Pre-calculate colors and fonts outside loop
        color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # Smaller font for speed
        thickness = 1  # Thinner lines for speed
        
        for pred in predictions:
            # Fast integer operations
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            x1, y1 = int(x - w // 2), int(y - h // 2)
            x2, y2 = int(x + w // 2), int(y + h // 2)

            # Minimal drawing operations
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Only draw label if confidence is high enough (skip low confidence)
            if pred["confidence"] > 0.5:
                label = f"{pred['class'][:8]} {pred['confidence']:.1f}"  # Shorter labels
                cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, color, thickness)
    
    except:
        pass  # Silent error handling for maximum speed
    
    return frame, predictions

# -------------------------
# Ultra-Fast Threaded Inference Worker
# -------------------------
class UltraFastInferenceWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame_queue = queue.Queue(maxsize=1)  # Single frame queue
        self.result_queue = queue.Queue(maxsize=3)
        self.running = True
        self.skip_count = 0
        
    def add_frame(self, frame):
        try:
            # Always clear queue and add latest frame only
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                
                # Process with ultra-fast method
                processed_frame, predictions = model_predict_ultra_fast(frame, force_inference=True)
                
                # Keep only latest result
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                
                try:
                    self.result_queue.put_nowait((processed_frame, predictions))
                except queue.Full:
                    pass
                
            except queue.Empty:
                continue
            except Exception:
                continue  # Silent error handling
    
    def stop(self):
        self.running = False

# -------------------------
# ADC Data Receiver Thread
# -------------------------
class ADCReceiver(threading.Thread):
    def __init__(self, udp_port=5006):
        super().__init__(daemon=True)
        self.udp_port = udp_port
        self.running = True
        self.latest_voltage = 0.0
        self.data_queue = queue.Queue(maxsize=100)
        
        # Setup UDP socket for ADC data
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("0.0.0.0", udp_port))
            self.sock.settimeout(0.1)  # Non-blocking with timeout
            print(f"ADC UDP socket bound to port {udp_port}")
        except Exception as e:
            print(f"ADC socket binding error: {e}")
    
    def get_latest_voltage(self):
        return self.latest_voltage
    
    def get_voltage_data(self):
        """Get all queued voltage data"""
        data = []
        try:
            while True:
                data.append(self.data_queue.get_nowait())
        except queue.Empty:
            pass
        return data
    
    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                json_str = data.decode('utf-8')
                adc_data = json.loads(json_str)
                
                # Update latest voltage
                self.latest_voltage = adc_data.get('voltage', 0.0)
                
                # Add to queue for plotting
                try:
                    self.data_queue.put_nowait({
                        'voltage': self.latest_voltage,
                        'timestamp': adc_data.get('timestamp', time.time())
                    })
                except queue.Full:
                    # Remove oldest data if queue is full
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put_nowait({
                            'voltage': self.latest_voltage,
                            'timestamp': adc_data.get('timestamp', time.time())
                        })
                    except:
                        pass
                        
            except socket.timeout:
                continue
            except Exception as e:
                continue  # Silent error handling
    
    def stop(self):
        self.running = False
        if hasattr(self, 'sock'):
            self.sock.close()



# -------------------------
# Optimized Tkinter GUI
# -------------------------
class UDPVideoApp:
    def __init__(self, root, udp_ip="0.0.0.0", video_port=5005, adc_port=5006):
        self.root = root
        self.root.title("UDP Video + ADC Display")
        self.root.geometry("1400x700")  # Larger window for bigger video display

        # Performance settings
        self.use_threading = tk.BooleanVar(value=True)
        self.skip_inference = tk.BooleanVar(value=False)
        self.ultra_mode = tk.BooleanVar(value=True)  # New ultra-fast mode
        
        # Setup Video UDP Socket with larger buffer
        self.video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536 * 4)
        try:
            self.video_sock.bind((udp_ip, video_port))
            self.video_sock.setblocking(False)
            print(f"Video UDP socket bound to {udp_ip}:{video_port}")
        except Exception as e:
            print(f"Video socket binding error: {e}")
        
        # Setup ADC receiver thread
        self.adc_receiver = ADCReceiver(adc_port)
        self.adc_receiver.start()

        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.display_fps = 0
        
        # Threading for inference
        self.inference_worker = UltraFastInferenceWorker()
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
        tk.Checkbutton(self.controls_frame, text="Skip Inference", 
                      variable=self.skip_inference).pack(side=tk.LEFT)
        tk.Checkbutton(self.controls_frame, text="Ultra Mode", 
                      variable=self.ultra_mode).pack(side=tk.LEFT)
        
        # Status labels
        self.status_label = tk.Label(self.video_frame, text="Waiting for UDP frames...", 
                                   font=("Arial", 10))
        self.status_label.pack()
        
        self.fps_label = tk.Label(self.video_frame, text="FPS: 0", 
                                font=("Arial", 10), fg="blue")
        self.fps_label.pack()
        
        # ADC status label
        self.adc_status_label = tk.Label(self.video_frame, text="ADC: Waiting...", 
                                       font=("Arial", 10), fg="green")
        self.adc_status_label.pack()

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
        
        # Fallback voltage toggle
        self.use_fallback_voltage = tk.BooleanVar()
        self.voltage_checkbox = tk.Checkbutton(self.video_frame, text="Use Simulated ADC (fallback)", 
                                             variable=self.use_fallback_voltage)
        self.voltage_checkbox.pack()
        
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

        # --- Voltage Graph Setup ---
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_ylim(0, 3.5)  # Adjusted for 3.3V range
        self.ax.set_title("ADC Voltage vs Time")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Voltage (V)")
        self.x_data = []
        self.y_data = []
        self.line, = self.ax.plot([], [], 'r-', linewidth=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Voltage update counter
        self.voltage_update_counter = 0
        self.plot_time_offset = time.time()

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
            data, addr = self.video_sock.recvfrom(65536)
            npdata = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.status_label.config(text=f"Video UDP: {addr[0]}:{addr[1]}")
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
                    processed_frame, _ = model_predict_ultra_fast(frame)
                
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

        # Update voltage plot
        self.voltage_update_counter += 1
        if self.voltage_update_counter >= 5:  # Update every 5 frames for smoother plotting
            self.voltage_update_counter = 0
            try:
                # Get voltage data from ADC receiver
                voltage_data_list = self.adc_receiver.get_voltage_data()
                
                if voltage_data_list:
                    # Process all received voltage data
                    for voltage_data in voltage_data_list:
                        voltage = voltage_data['voltage']
                        
                        # Add to plot data
                        current_time = time.time() - self.plot_time_offset
                        if len(self.x_data) >= 100:  # Keep last 100 points
                            self.x_data = self.x_data[1:]
                            self.y_data = self.y_data[1:]
                        
                        self.x_data.append(current_time)
                        self.y_data.append(voltage)
                    
                    # Update status with latest voltage
                    latest_voltage = self.adc_receiver.get_latest_voltage()
                    self.adc_status_label.config(text=f"ADC: {latest_voltage:.3f}V (UDP)")
                    
                    # Update plot
                    if len(self.x_data) > 0:
                        self.line.set_data(self.x_data, self.y_data)
                        if len(self.x_data) > 1:
                            self.ax.set_xlim(min(self.x_data), max(self.x_data))
                        self.canvas.draw_idle()  # Use draw_idle for better performance
                else:
                    # No new UDP data, just update status
                    latest_voltage = self.adc_receiver.get_latest_voltage()
                    if latest_voltage > 0:
                        self.adc_status_label.config(text=f"ADC: {latest_voltage:.3f}V (UDP)")
                    else:
                        self.adc_status_label.config(text="ADC: Waiting for UDP data...")
                    
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
        if hasattr(self, 'adc_receiver'):
            self.adc_receiver.stop()
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'video_sock'):
            self.video_sock.close()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = UDPVideoApp(root, video_port=5005, adc_port=5006)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
    finally:
        if hasattr(app, 'inference_worker'):
            app.inference_worker.stop()
        if hasattr(app, 'adc_receiver'):
            app.adc_receiver.stop()
        if hasattr(app, 'cap') and app.cap is not None:
            app.cap.release()
        if hasattr(app, 'video_sock'):
            app.video_sock.close()