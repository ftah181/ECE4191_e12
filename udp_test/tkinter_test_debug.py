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

# --- Setup Roboflow client ---
model = get_model("animal-detection-evlon/1", api_key="lnHqcMh4NynT1If5FC38")

# Global variables for performance optimization
frame_skip_counter = 0
INFERENCE_SKIP_FRAMES = 10  # Reduced for more frequent inference
last_predictions = []
frame_buffer = None

# Debug flag
DEBUG_INFERENCE = True

# -------------------------
# Improved Model Prediction with Debug
# -------------------------
def model_predict_debug(frame, force_inference=False):
    """
    Model prediction with extensive debugging
    """
    global frame_skip_counter, last_predictions, frame_buffer
    
    if DEBUG_INFERENCE:
        print(f"\n=== INFERENCE DEBUG ===")
        print(f"Frame shape: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")
        print(f"Frame skip counter: {frame_skip_counter}")
        print(f"Force inference: {force_inference}")
    
    # Skip inference logic
    frame_skip_counter += 1
    if not force_inference and frame_skip_counter < INFERENCE_SKIP_FRAMES:
        if DEBUG_INFERENCE:
            print(f"Skipping inference, using cached predictions: {len(last_predictions)} predictions")
        return draw_predictions_debug(frame, last_predictions)
    
    frame_skip_counter = 0
    
    # Store original frame info
    original_shape = frame.shape[:2]
    if DEBUG_INFERENCE:
        print(f"Original frame size: {original_shape}")
    
    # Use larger inference size for better detection
    inference_size = (640, 480)  # Increased from 160x120
    resized_frame = cv2.resize(frame, inference_size)
    
    if DEBUG_INFERENCE:
        print(f"Resized frame to: {resized_frame.shape}")
    
    predictions = []
    
    try:
        # Convert BGR to RGB properly
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        if DEBUG_INFERENCE:
            print(f"PIL image mode: {pil_image.mode}")
            print(f"PIL image size: {pil_image.size}")
            print("Running inference...")
        
        # Run inference with error handling
        result = model.predict(pil_image)
        
        if DEBUG_INFERENCE:
            print(f"Raw inference result: {result}")
            print(f"Result type: {type(result)}")
        
        # Handle different result formats
        if hasattr(result, 'predictions'):
            predictions_data = result.predictions
        elif isinstance(result, dict) and 'predictions' in result:
            predictions_data = result['predictions']
        elif isinstance(result, list):
            predictions_data = result
        else:
            if DEBUG_INFERENCE:
                print(f"Unexpected result format: {result}")
            predictions_data = []
        
        if DEBUG_INFERENCE:
            print(f"Found {len(predictions_data)} predictions")
        
        # Scale predictions back to original size
        scale_x = original_shape[1] / inference_size[0]
        scale_y = original_shape[0] / inference_size[1]
        
        if DEBUG_INFERENCE:
            print(f"Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
        
        predictions = []
        for i, pred in enumerate(predictions_data):
            if DEBUG_INFERENCE:
                print(f"Processing prediction {i}: {pred}")
            
            # Handle different prediction formats
            try:
                if hasattr(pred, 'class_name'):
                    class_name = pred.class_name
                elif hasattr(pred, 'class_id'):
                    class_name = getattr(pred, 'class_id', 'unknown')
                elif isinstance(pred, dict):
                    class_name = pred.get('class', pred.get('class_name', 'unknown'))
                else:
                    # Handle the case where pred has a 'class' attribute (reserved keyword)
                    try:
                        class_name = getattr(pred, 'class', 'unknown')
                    except:
                        class_name = 'unknown'
                
                if hasattr(pred, 'confidence'):
                    confidence = pred.confidence
                elif isinstance(pred, dict):
                    confidence = pred.get('confidence', 0.0)
                else:
                    confidence = 0.0
                
                # Get bounding box coordinates
                if hasattr(pred, 'x') and hasattr(pred, 'y'):
                    x, y = pred.x, pred.y
                    width = getattr(pred, 'width', getattr(pred, 'w', 50))
                    height = getattr(pred, 'height', getattr(pred, 'h', 50))
                elif isinstance(pred, dict):
                    x = pred.get('x', pred.get('center_x', 0))
                    y = pred.get('y', pred.get('center_y', 0))
                    width = pred.get('width', pred.get('w', 50))
                    height = pred.get('height', pred.get('h', 50))
                else:
                    x, y, width, height = 100, 100, 50, 50  # Default values
                
                # Scale to original image size
                scaled_pred = {
                    "class": class_name,
                    "confidence": float(confidence),
                    "x": int(x * scale_x),
                    "y": int(y * scale_y),
                    "width": int(width * scale_x),
                    "height": int(height * scale_y)
                }
                
                predictions.append(scaled_pred)
                
                if DEBUG_INFERENCE:
                    print(f"Scaled prediction {i}: {scaled_pred}")
                    
            except Exception as pred_error:
                print(f"Error processing prediction {i}: {pred_error}")
                continue
        
        # Cache predictions
        last_predictions = predictions
        
        if DEBUG_INFERENCE:
            print(f"Final predictions count: {len(predictions)}")
            print("=== END INFERENCE DEBUG ===\n")
        
    except Exception as inference_error:
        print(f"INFERENCE ERROR: {inference_error}")
        import traceback
        traceback.print_exc()
        predictions = last_predictions  # Use cached predictions
    
    return draw_predictions_debug(frame, predictions)

def draw_predictions_debug(frame, predictions):
    """
    Draw predictions with debug info
    """
    if DEBUG_INFERENCE and predictions:
        print(f"Drawing {len(predictions)} predictions")
    
    try:
        # Colors for different classes
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        for i, pred in enumerate(predictions):
            try:
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                confidence = pred["confidence"]
                class_name = pred["class"]
                
                # Calculate bounding box corners
                x1, y1 = int(x - w // 2), int(y - h // 2)
                x2, y2 = int(x + w // 2), int(y + h // 2)
                
                # Ensure coordinates are within frame bounds
                frame_h, frame_w = frame.shape[:2]
                x1 = max(0, min(x1, frame_w - 1))
                y1 = max(0, min(y1, frame_h - 1))
                x2 = max(0, min(x2, frame_w - 1))
                y2 = max(0, min(y2, frame_h - 1))
                
                # Choose color
                color = colors[i % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label with confidence
                label = f"{class_name} {confidence:.2f}"
                label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                
                # Draw label background
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, 
                           (255, 255, 255), thickness)
                
                if DEBUG_INFERENCE:
                    print(f"Drew prediction: {label} at ({x1},{y1})-({x2},{y2})")
                    
            except Exception as draw_error:
                print(f"Error drawing prediction {i}: {draw_error}")
                continue
    
    except Exception as e:
        print(f"Error in draw_predictions_debug: {e}")
    
    return frame, predictions

# -------------------------
# Fallback Test Function
# -------------------------
def test_model_with_sample():
    """
    Test the model with a sample image
    """
    print("\n=== TESTING MODEL WITH SAMPLE ===")
    
    # Create a test image
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Or use a solid color image
    test_frame[:] = [100, 150, 200]  # Light blue
    
    # Add some patterns to make it more interesting
    cv2.rectangle(test_frame, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(test_frame, (400, 300), 50, (0, 0, 0), -1)
    
    print("Testing model with synthetic image...")
    result_frame, predictions = model_predict_debug(test_frame, force_inference=True)
    
    print(f"Test completed. Found {len(predictions)} predictions")
    return result_frame, predictions

# -------------------------
# Enhanced Threading Worker
# -------------------------
class EnhancedInferenceWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=3)
        self.running = True
        self.processed_count = 0
        
    def add_frame(self, frame):
        try:
            # Clear old frames
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.frame_queue.put_nowait(frame.copy())  # Make a copy to avoid issues
        except queue.Full:
            pass
    
    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def run(self):
        print("Inference worker thread started")
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                
                if DEBUG_INFERENCE:
                    print(f"\nWorker processing frame {self.processed_count}")
                
                processed_frame, predictions = model_predict_debug(frame, force_inference=True)
                self.processed_count += 1
                
                # Clear old results
                while not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        break
                
                try:
                    self.result_queue.put_nowait((processed_frame, predictions))
                except queue.Full:
                    pass
                
                if DEBUG_INFERENCE:
                    print(f"Worker completed frame {self.processed_count}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                import traceback
                traceback.print_exc()
    
    def stop(self):
        print("Stopping inference worker")
        self.running = False

# -------------------------
# ADC Data Receiver (unchanged)
# -------------------------
class ADCReceiver(threading.Thread):
    def __init__(self, udp_port=5006):
        super().__init__(daemon=True)
        self.udp_port = udp_port
        self.running = True
        self.latest_voltage = 0.0
        self.data_queue = queue.Queue(maxsize=100)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("0.0.0.0", udp_port))
            self.sock.settimeout(0.1)
            print(f"ADC UDP socket bound to port {udp_port}")
        except Exception as e:
            print(f"ADC socket binding error: {e}")
    
    def get_latest_voltage(self):
        return self.latest_voltage
    
    def get_voltage_data(self):
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
                
                self.latest_voltage = adc_data.get('voltage', 0.0)
                
                try:
                    self.data_queue.put_nowait({
                        'voltage': self.latest_voltage,
                        'timestamp': adc_data.get('timestamp', time.time())
                    })
                except queue.Full:
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
            except Exception:
                continue
    
    def stop(self):
        self.running = False
        if hasattr(self, 'sock'):
            self.sock.close()

# -------------------------
# Enhanced GUI
# -------------------------
class EnhancedUDPVideoApp:
    def __init__(self, root, udp_ip="0.0.0.0", video_port=5005, adc_port=5006):
        self.root = root
        self.root.title("Enhanced UDP Video + ADC Display")
        self.root.geometry("1600x800")

        # Performance settings
        self.use_threading = tk.BooleanVar(value=True)
        self.skip_inference = tk.BooleanVar(value=False)
        self.debug_mode = tk.BooleanVar(value=True)
        
        # Setup Video UDP Socket
        self.video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536 * 4)
        try:
            self.video_sock.bind((udp_ip, video_port))
            self.video_sock.setblocking(False)
            print(f"Video UDP socket bound to {udp_ip}:{video_port}")
        except Exception as e:
            print(f"Video socket binding error: {e}")
        
        # Setup ADC receiver
        self.adc_receiver = ADCReceiver(adc_port)
        self.adc_receiver.start()

        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.display_fps = 0
        
        # Enhanced threading for inference
        self.inference_worker = EnhancedInferenceWorker()
        self.inference_worker.start()
        self.last_inference_result = None
        
        self.setup_ui()
        
        # Initialize webcam
        self.cap = None
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                print("Webcam initialized successfully")
            else:
                self.cap = None
                print("Could not open webcam")
        except Exception as e:
            print(f"Webcam initialization error: {e}")
            self.cap = None

        self.frame_count = 0
        
        # Test the model on startup
        self.test_model_button_click()
        
        # Start update loop
        self.update()
    
    def setup_ui(self):
        # Main frame layout
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - video and controls
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Controls frame
        self.controls_frame = tk.Frame(self.left_frame)
        self.controls_frame.pack(fill=tk.X)
        
        # Control buttons and checkboxes
        controls_row1 = tk.Frame(self.controls_frame)
        controls_row1.pack(fill=tk.X)
        
        tk.Checkbutton(controls_row1, text="Threaded Inference", 
                      variable=self.use_threading).pack(side=tk.LEFT)
        tk.Checkbutton(controls_row1, text="Skip Inference", 
                      variable=self.skip_inference).pack(side=tk.LEFT)
        tk.Checkbutton(controls_row1, text="Debug Mode", 
                      variable=self.debug_mode,
                      command=self.toggle_debug).pack(side=tk.LEFT)
        
        # Test button
        self.test_button = tk.Button(controls_row1, text="Test Model", 
                                    command=self.test_model_button_click,
                                    bg="lightgreen")
        self.test_button.pack(side=tk.LEFT, padx=5)
        
        # Status labels
        self.status_label = tk.Label(self.left_frame, text="Waiting for UDP frames...", 
                                   font=("Arial", 10))
        self.status_label.pack()
        
        self.fps_label = tk.Label(self.left_frame, text="FPS: 0", 
                                font=("Arial", 10), fg="blue")
        self.fps_label.pack()
        
        self.inference_status_label = tk.Label(self.left_frame, text="Inference: Ready", 
                                             font=("Arial", 10), fg="purple")
        self.inference_status_label.pack()
        
        self.adc_status_label = tk.Label(self.left_frame, text="ADC: Waiting...", 
                                       font=("Arial", 10), fg="green")
        self.adc_status_label.pack()

        # Video display
        self.video_label = tk.Label(self.left_frame, bg="black", text="No video")
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Frame counter and testing options
        self.frame_counter_label = tk.Label(self.left_frame, text="Frames: 0")
        self.frame_counter_label.pack()
        
        self.use_webcam = tk.BooleanVar()
        tk.Checkbutton(self.left_frame, text="Use Webcam for Testing", 
                      variable=self.use_webcam).pack()
        
        # Right side - voltage graph
        self.setup_voltage_graph()
    
    def setup_voltage_graph(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_ylim(0, 3.5)
        self.ax.set_title("ADC Voltage vs Time")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Voltage (V)")
        self.x_data = []
        self.y_data = []
        self.line, = self.ax.plot([], [], 'r-', linewidth=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.voltage_update_counter = 0
        self.plot_time_offset = time.time()
    
    def toggle_debug(self):
        global DEBUG_INFERENCE
        DEBUG_INFERENCE = self.debug_mode.get()
        print(f"Debug mode: {'ON' if DEBUG_INFERENCE else 'OFF'}")
    
    def test_model_button_click(self):
        """Test button click handler"""
        print("Testing model...")
        self.inference_status_label.config(text="Inference: Testing...")
        
        try:
            result_frame, predictions = test_model_with_sample()
            self.inference_status_label.config(
                text=f"Inference: Test OK ({len(predictions)} predictions)")
        except Exception as e:
            print(f"Model test failed: {e}")
            self.inference_status_label.config(text=f"Inference: Test FAILED")
    
    def calculate_fps(self):
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.display_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            self.fps_label.config(text=f"FPS: {self.display_fps:.1f}")
    
    def update(self):
        frame = None
        
        # Try UDP first
        try:
            data, addr = self.video_sock.recvfrom(65536)
            npdata = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.status_label.config(text=f"Video UDP: {addr[0]}:{addr[1]}")
                self.frame_count += 1
        except socket.error:
            pass
        
        # Fallback to webcam
        if frame is None and self.use_webcam.get() and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.status_label.config(text="Using webcam")
                self.frame_count += 1
        
        # Process frame
        if frame is not None:
            try:
                if self.skip_inference.get():
                    processed_frame = frame
                    self.inference_status_label.config(text="Inference: SKIPPED")
                elif self.use_threading.get():
                    # Threaded inference
                    self.inference_worker.add_frame(frame)
                    
                    result = self.inference_worker.get_result()
                    if result is not None:
                        self.last_inference_result = result
                        processed_frame, predictions = result
                        self.inference_status_label.config(
                            text=f"Inference: {len(predictions)} predictions")
                    else:
                        if self.last_inference_result is not None:
                            processed_frame, predictions = self.last_inference_result
                        else:
                            processed_frame = frame
                            self.inference_status_label.config(text="Inference: Processing...")
                else:
                    # Direct inference
                    processed_frame, predictions = model_predict_debug(frame)
                    self.inference_status_label.config(
                        text=f"Inference: {len(predictions)} predictions")
                
                # Display frame
                display_frame = self.resize_for_display(processed_frame)
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                self.frame_counter_label.config(text=f"Frames: {self.frame_count}")
                self.calculate_fps()
                
            except Exception as e:
                print(f"Frame processing error: {e}")
                import traceback
                traceback.print_exc()

        # Update voltage plot (simplified)
        self.update_voltage_plot()
        
        # Continue loop
        self.root.after(33, self.update)  # ~30 FPS
    
    def update_voltage_plot(self):
        self.voltage_update_counter += 1
        if self.voltage_update_counter >= 5:
            self.voltage_update_counter = 0
            try:
                voltage_data_list = self.adc_receiver.get_voltage_data()
                
                if voltage_data_list:
                    for voltage_data in voltage_data_list:
                        voltage = voltage_data['voltage']
                        current_time = time.time() - self.plot_time_offset
                        
                        if len(self.x_data) >= 100:
                            self.x_data = self.x_data[1:]
                            self.y_data = self.y_data[1:]
                        
                        self.x_data.append(current_time)
                        self.y_data.append(voltage)
                    
                    latest_voltage = self.adc_receiver.get_latest_voltage()
                    self.adc_status_label.config(text=f"ADC: {latest_voltage:.3f}V")
                    
                    if len(self.x_data) > 0:
                        self.line.set_data(self.x_data, self.y_data)
                        if len(self.x_data) > 1:
                            self.ax.set_xlim(min(self.x_data), max(self.x_data))
                        self.canvas.draw_idle()
            except Exception as e:
                if DEBUG_INFERENCE:
                    print(f"Plot update error: {e}")
    
    def resize_for_display(self, frame):
        height, width = frame.shape[:2]
        max_width, max_height = 800, 600
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
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
# Main
# -------------------------
if __name__ == "__main__":
    print("Starting Enhanced UDP Video App with Debug")
    root = tk.Tk()
    app = EnhancedUDPVideoApp(root, video_port=5005, adc_port=5006)
    
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