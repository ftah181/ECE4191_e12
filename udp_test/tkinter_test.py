import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import socket
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time
import json
from inference import get_model
from ultralytics import YOLO

# Load model
#model = get_model("animal-detection-evlon/3", api_key="lnHqcMh4NynT1If5FC38")
model = YOLO("models/runs/train/my_model/weights/best.pt")
# model = YOLO("models/yolo11n.pt")

# Global variables
frame_skip_counter = 0
last_predictions = []  # Cache last predictions
frame_buffer = None  # Buffer for frame reuse

# Params
INFERENCE_SKIP_FRAMES = 30  # Run inference every N frames
CONF_THRESHOLD = 0.7 # Confidence threshold for predictions
SAMPLE_RATE = 1000 # Sample rate of ADC data for spectrogram
BUFFER_SIZE = 2048
PLOT_X_LENGTH = 1024


# -------------------------
#  Run Inference Model
# -------------------------
def model_predict(frame, force_inference=False):
    global frame_skip_counter, last_predictions

    frame_skip_counter += 1
    if not force_inference and frame_skip_counter < INFERENCE_SKIP_FRAMES:
        return draw_predictions(frame, last_predictions)

    frame_skip_counter = 0
    predictions = []

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, 
                        imgsz=640,           # Increase size
                        conf=0.25,           # Confidence threshold
                        iou=0.45,            # NMS IoU threshold
                        max_det=300,         # Max detections
                        augment=False,       # Test time augmentation
                        agnostic_nms=False)  # Class-agnostic NMS

        res = results[0]
        predictions = []
        h, w, _ = frame.shape

        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            #print(f"YOLO box: {x1},{y1},{x2},{y2}")

            confidence = float(box.conf[0].item())
            cls = int(box.cls[0].item())
            class_name = res.names[cls]

            prediction_dict = {
                "class": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
            predictions.append(prediction_dict)

        last_predictions = predictions

    except Exception as e:
        print(f"Inference error: {e}")
        predictions = last_predictions

    return draw_predictions(frame, predictions)


# -------------------------
#  Overlay Bounding Boxes
# -------------------------
def draw_predictions(frame, predictions):
    try:
        color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        for pred in predictions:
            if pred["confidence"] > CONF_THRESHOLD:
                x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Label
                label = f"{pred['class'][:8]} {pred['confidence']:.1f}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            font, font_scale, color, thickness)

    except Exception as e:
        print(f"Draw error: {e}")

    return frame, predictions


# -------------------------
# Threaded Inference Worker
# -------------------------
class InferenceWorker(threading.Thread):
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
                
                # Run inference
                processed_frame, predictions = model_predict(frame, force_inference=True)
                
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
                continue
    
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
        self.data_queue = queue.Queue(maxsize=1024)
        
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
                print("running")

                data, addr = self.sock.recvfrom(4096)
                json_str = data.decode('utf-8')
                adc_data = json.loads(json_str)

                # Expect adc_data like: {"voltages": [...], "timestamp": ...}
                voltages = adc_data.get("voltages", [])
                timestamp = adc_data.get("timestamp", time.time())

                if voltages:
                    # Latest voltage is the last one in the list
                    self.latest_voltage = voltages[-1]

                    # Add each voltage to queue with timestamp
                    for v in voltages:
                        entry = {"voltage": v, "timestamp": timestamp}
                        try:
                            self.data_queue.put_nowait(entry)
                        except queue.Full:
                            # Remove oldest if queue full
                            try:
                                self.data_queue.get_nowait()
                                self.data_queue.put_nowait(entry)
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
# Video Receiving Thread
# -------------------------
class VideoReceiver(threading.Thread):
    def __init__(self, udp_ip, video_port, max_queue=3):
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

# -------------------------
# Tkinter GUI
# -------------------------
class GUI:
    def __init__(self, root, udp_ip="0.0.0.0", video_port=5005, adc_port=5006):
        self.root = root
        self.root.title("UDP Video + ADC Display")
        self.root.geometry("1400x800")  # Adjusted height for new layout
        self.root.configure(bg='#2C3E50')

        # Performance settings
        self.skip_inference = tk.BooleanVar(value=False)
        
        # Setup video receiver thread
        self.video_receiver = VideoReceiver(udp_ip, video_port)
        self.video_receiver.start()
        
        # Setup ADC receiver thread
        self.adc_receiver = ADCReceiver(adc_port)
        self.adc_receiver.start()

        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.display_fps = 0
        
        # Threading for inference (always enabled)
        self.inference_worker = InferenceWorker()
        self.inference_worker.start()
        self.last_inference_result = None
        
        # Main layout: Left side for video and graph, right side reserved
        self.left_frame = tk.Frame(root, bg='#2C3E50')
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        self.right_frame = tk.Frame(root, bg='#34495E', width=300)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.right_frame.pack_propagate(False)  # Maintain fixed width
        
        # Animal detection list
        self.detection_title = tk.Label(self.right_frame, text="Detected Animals", 
                                       font=("Arial", 14, "bold"), bg='#34495E', fg='white')
        self.detection_title.pack(pady=(10, 5))
        
        # Scrollable frame for animal list
        self.detection_section = tk.Frame(self.right_frame, bg='#34495E')
        self.detection_section.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=(0, 10))
        self.detection_canvas = tk.Canvas(self.detection_section, bg='#34495E', highlightthickness=0)
        self.detection_scrollbar = tk.Scrollbar(self.detection_section, orient="vertical", command=self.detection_canvas.yview)
        self.detection_scrollable_frame = tk.Frame(self.detection_canvas, bg='#34495E')

        self.detection_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.detection_canvas.configure(scrollregion=self.detection_canvas.bbox("all"))
        )
        
        self.detection_canvas.create_window((0, 0), window=self.detection_scrollable_frame, anchor="nw")
        self.detection_canvas.configure(yscrollcommand=self.detection_scrollbar.set)
        
        self.detection_canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=5)
        self.detection_scrollbar.pack(side="right", fill="y", pady=5)
        
        # Clear button
        self.clear_button = tk.Button(self.detection_section, text="Clear List", 
                                     command=self.clear_detection_list,
                                     bg='#E74C3C', fg='white', font=("Arial", 10))
        self.clear_button.pack(pady=5)
        
        # Detection tracking
        self.detection_list = []
        self.detection_counter = 0
        self.detected_animals = set()  # Track unique animals
        
        # Top left: Video frame
        self.video_frame = tk.Frame(self.left_frame, bg='#2C3E50')
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=(0, 10))
        
        # Performance controls
        self.controls_frame = tk.Frame(self.video_frame)
        self.controls_frame.pack()
        
        tk.Checkbutton(self.controls_frame, text="Skip Inference", 
                      variable=self.skip_inference).pack(side=tk.LEFT)
        
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
        
        # Bottom left: Voltage Graph
        self.graph_frame = tk.Frame(self.left_frame, bg='#2C3E50')
        self.graph_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Bottom Right: Spectrogram
        self.spec_frame = tk.Frame(self.right_frame, bg='#2C3E50', height=250)
        self.spec_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False)

        # Voltage Graph Setup
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.fig.patch.set_facecolor('#2C3E50')
        self.ax.set_facecolor('#34495E')
        self.ax.set_ylim(0, 5)
        self.ax.set_title("Microphone Data", color='white')
        self.ax.set_xlabel("Time", color='white')
        self.ax.set_ylabel("Magnitude", color='white')
        self.ax.tick_params(colors='white')
        self.x_data = []
        self.y_data = []
        self.time_window = 10  
        self.line, = self.ax.plot([], [], 'r-', linewidth=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Spectogram setup
        self.fig_spec, self.ax_spec = plt.subplots(figsize=(6, 3))
        self.ax_spec.set_title("Spectrogram")
        self.ax_spec.set_xlabel("Time [s]")
        self.ax_spec.set_ylabel("Frequency [Hz]")
        self.spectogram_queue = []
        self.canvas_spec = FigureCanvasTkAgg(self.fig_spec, master=self.spec_frame)
        self.canvas_spec.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Voltage update counter
        self.voltage_update_counter = 0
        self.plot_time_offset = time.time()

        # Setup VideoWriter
        self.video_out = None
        self.recording = False
        self.video_filename = "output.mp4"

        # Recording button
        self.record_button = tk.Button(root, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=10)

        self.update()  # start loop

    def toggle_recording(self):
        if not self.recording:
            # Start recording
            self.recording = True
            self.record_button.config(text="Stop Recording")
            print("Recording started")
        else:
            # Stop recording
            self.recording = False
            self.record_button.config(text="Start Recording")
            if self.video_out is not None:
                self.video_out.release()
                self.video_out = None
            print("Recording stopped")
    
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
        frames_processed = 0

        # Use non-blocking receive
        new_frame = self.video_receiver.get_latest_frame()
        
        if new_frame is not None:
            # Always use the latest frame (drop older ones)
            frame = new_frame
            frames_processed += 1
            self.frame_count += 1
            
            # Update status only for the latest frame
            if frames_processed == 1:  # Only update status once
                self.status_label.config(text=f"Video UDP:")   
                    
        
        # Process the latest frame if available
        if frame is not None:
            try:
                # Skip inference if requested (display only mode)
                if self.skip_inference.get():
                    processed_frame = frame
                else:
                    # Pass frame to inference thread
                    self.inference_worker.add_frame(frame)
                    
                    # Check for completed inference
                    result = self.inference_worker.get_result()
                    if result is not None:
                        self.last_inference_result = result

                        # Update detection list with new predictions
                        _, predictions = result
                        self.update_detection_list(predictions)
                    
                    # Use last result or original frame
                    if self.last_inference_result is not None:
                        processed_frame, _ = self.last_inference_result
                    else:
                        processed_frame = frame
                
                # Resize for display if too large
                display_frame = self.resize_for_display(processed_frame)
                
                # Convert and display
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                if self.recording:
                    if self.video_out is None:
                        # Define the codec and create VideoWriter object
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
                        self.video_out = cv2.VideoWriter(self.video_filename, fourcc, 20.0, (w, h))
                    
                        if not self.video_out.isOpened():
                            print("Failed to initialize VideoWriter")
                            self.video_out = None
                
                    # Record frames
                    if self.video_out is not None:
                        self.video_out.write(display_frame)

                # Update counters
                self.frame_counter_label.config(text=f"Frames: {self.frame_count}")
                self.calculate_fps()
                
            except Exception as e:
                print(f"Frame processing error: {e}")

        # Update voltage plot (less frequently to reduce overhead)
        self.voltage_update_counter += 1
        if self.voltage_update_counter >= 10:  # Reduced frequency for better performance
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
                        if len(self.x_data) >= PLOT_X_LENGTH:
                            self.x_data = self.x_data[1:]
                            self.y_data = self.y_data[1:]
                        
                        self.x_data.append(current_time)
                        self.y_data.append(voltage)
                    
                    # Update status with latest voltage
                    latest_voltage = self.adc_receiver.get_latest_voltage()
                    self.adc_status_label.config(text=f"ADC: {latest_voltage:.3f}V (UDP)")
                    
                    # Update spectrogram
                    Pxx, freqs, bins, im = self.ax_spec.specgram(self.y_data, NFFT=1024, Fs=SAMPLE_RATE, noverlap=512, cmap="viridis")
                    self.canvas_spec.draw()

                    # Update plot
                    if len(self.x_data) > 0:
                        # Keep only data within time_window
                        while self.x_data and (self.x_data[-1] - self.x_data[0]) > self.time_window:
                            self.x_data.pop(0)
                            self.y_data.pop(0)

                        self.line.set_data(self.x_data, self.y_data)
                        if len(self.x_data) > 1:
                            self.ax.set_xlim(self.x_data[0], self.x_data[0] + self.time_window)

                        self.canvas.draw_idle()
                else:
                    # No new UDP data, just update status
                    latest_voltage = self.adc_receiver.get_latest_voltage()
                    if latest_voltage > 0:
                        self.adc_status_label.config(text=f"ADC: {latest_voltage:.3f}V (UDP)")
                    else:
                        self.adc_status_label.config(text="ADC: Waiting for UDP data...")
                    
            except Exception as e:
                print(f"Plot update error: {e}")

        # Faster update cycle - consider making this adaptive
        self.root.after(5, self.update)  # Increase frequency to clear buffer faster
    
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
    
    def update_detection_list(self, predictions):
        """Update the animal detection list with new predictions - only unique animals"""
        import datetime
        
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        
        for pred in predictions:
            if pred["confidence"] > 0.5:  # Only add high-confidence detections
                animal_name = pred["class"].lower()  # Convert to lowercase for comparison
                
                # Only add if this animal hasn't been detected before
                if animal_name not in self.detected_animals:
                    self.detected_animals.add(animal_name)
                    self.detection_counter += 1
                    confidence = pred["confidence"]
                    
                    # Create detection entry
                    detection_frame = tk.Frame(self.detection_scrollable_frame, bg='#2C3E50', 
                                             relief=tk.RAISED, bd=1)
                    detection_frame.pack(fill=tk.X, padx=5, pady=2)
                    
                    # Detection info
                    info_text = f"#{self.detection_counter}: {pred['class']}\nConfidence: {confidence:.2f}\nTime: {current_time}"
                    detection_label = tk.Label(detection_frame, text=info_text, 
                                             bg='#2C3E50', fg='white', font=("Arial", 9),
                                             justify=tk.LEFT)
                    detection_label.pack(padx=5, pady=3)
                    
                    # Store detection info
                    self.detection_list.append({
                        'frame': detection_frame,
                        'animal': pred['class'],
                        'confidence': confidence,
                        'time': current_time
                    })
                    
                    # Auto-scroll to bottom
                    self.detection_canvas.update_idletasks()
                    self.detection_canvas.yview_moveto(1.0)
    
    def clear_detection_list(self):
        """Clear all detections from the list"""
        for detection in self.detection_list:
            detection['frame'].destroy()
        self.detection_list.clear()
        self.detection_counter = 0

    def __del__(self):
        if hasattr(self, 'inference_worker'):
            self.inference_worker.stop()
        if hasattr(self, 'adc_receiver'):
            self.adc_receiver.stop()
        if hasattr(self, 'video_sock'):
            self.video_sock.close()
        if hasattr(self, 'video_receiver'):
            self.video_receiver.stop()
        if self.video_out is not None:
            self.video_out.release()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root, video_port=5005, adc_port=5006)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted")
    finally:
        if hasattr(app, 'inference_worker'):
            app.inference_worker.stop()
        if hasattr(app, 'adc_receiver'):
            app.adc_receiver.stop()
        if hasattr(app, 'video_sock'):
            app.video_sock.close()