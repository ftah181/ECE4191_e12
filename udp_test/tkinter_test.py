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
from ultralytics import YOLO
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import pickle
import wave

# Load model
# model = YOLO("models/runs/train/my_model/weights/best.pt")
model = YOLO("Model_HL_16-09.pt")

# Load audio classification models
try:
    # Load label encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    # Load trained classifier
    audio_classifier = load_model("yamnet_audio_classifier_old.h5")
    
    # Load YAMNet
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    
    print("Audio classification models loaded successfully")
except Exception as e:
    print(f"Error loading audio models: {e}")
    label_encoder = None
    audio_classifier = None
    yamnet_model = None

# Global variables
frame_skip_counter = 0
last_predictions = []  # Cache last predictions
frame_buffer = None  # Buffer for frame reuse

# Params
INFERENCE_SKIP_FRAMES = 30      # Run inference every N frames
CONF_THRESHOLD = 0.7            # Confidence threshold for predictions
PLOT_X_LENGTH = 10000      
AUDIO_SAMPLES = 1024*200         # ~ 13 seconds worth of data

# Audio classification params
AUDIO_CLASSIFICATION_INTERVAL = 5.0     # Run audio classification every N seconds
AUDIO_CLASSIFICATION_CONFIDENCE = 0.5   # Minimum confidence for audio predictions
YAMNET_SAMPLE_RATE = 16000              # YAMNet expects 16kHz audio

# Audio recording params
AUDIO_SAMPLE_RATE = 16000               # Audio recording sample rate
AUDIO_CHUNK_SIZE = 16000                # Save audio in 1-second chunks


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
# Audio Classification Functions
# -------------------------
def extract_embedding(waveform):
    """Extract YAMNet embedding from audio waveform"""
    if yamnet_model is None:
        return None
    
    try:
        waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
        scores, embeddings, spec = yamnet_model(waveform)
        return tf.reduce_mean(embeddings, axis=0).numpy()
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

def classify_audio(audio_data):
    """Classify audio data using the trained model"""
    if audio_classifier is None or label_encoder is None:
        return None, 0.0
    
    try:
        # Convert to numpy array if it's a list
        if isinstance(audio_data, list):
            audio_np = np.array(audio_data, dtype=np.float32)
        else:
            audio_np = audio_data.astype(np.float32)
        
        # Resample to 16kHz if needed (YAMNet requirement)
        if len(audio_np) > 0:
            # Simple resampling - for better quality, use librosa.resample
            if len(audio_np) != YAMNET_SAMPLE_RATE:
                # Basic linear interpolation resampling
                indices = np.linspace(0, len(audio_np) - 1, YAMNET_SAMPLE_RATE)
                audio_np = np.interp(indices, np.arange(len(audio_np)), audio_np)
        
        # Extract embedding
        embedding = extract_embedding(audio_np)
        if embedding is None:
            return None, 0.0
        
        # Reshape for prediction
        embedding = embedding.reshape(1, -1)
        
        # Predict
        prediction = audio_classifier.predict(embedding, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class_idx])
        
        # Decode class name
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error classifying audio: {e}")
        return None, 0.0

# -------------------------
# Audio Classification Worker Thread
# -------------------------
class AudioClassificationWorker(threading.Thread):
    def __init__(self, adc_receiver):
        super().__init__(daemon=True)
        self.adc_receiver = adc_receiver
        self.running = True
        self.last_classification_time = 0
        self.result_queue = queue.Queue(maxsize=5)
        
    def get_result(self):
        """Get latest audio classification result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def run(self):
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for classification
                if current_time - self.last_classification_time >= AUDIO_CLASSIFICATION_INTERVAL:
                    # Get audio data from ADC receiver
                    audio_data = self.adc_receiver.get_voltage_data()
                    
                    if len(audio_data) > 0:
                        # Extract voltage values for classification
                        voltage_values = [entry['voltage'] for entry in audio_data]
                        
                        # Classify audio
                        predicted_class, confidence = classify_audio(voltage_values)
                        
                        if predicted_class is not None and confidence >= AUDIO_CLASSIFICATION_CONFIDENCE:
                            # Add result to queue
                            result = {
                                'class': predicted_class,
                                'confidence': confidence,
                                'timestamp': current_time
                            }
                            
                            # Keep only latest results
                            try:
                                while not self.result_queue.empty():
                                    self.result_queue.get_nowait()
                            except queue.Empty:
                                pass
                            
                            try:
                                self.result_queue.put_nowait(result)
                            except queue.Full:
                                pass
                            #print(result)
                            print(f"Audio classification: {predicted_class} (confidence: {confidence:.3f})")
                    
                    # Clear ADC data after classification
                    #self.adc_receiver.clear_voltage_data()

                    self.last_classification_time = current_time
                
                # Sleep to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Audio classification error: {e}")
                time.sleep(1.0)
    
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
        self.data_queue = queue.Queue(maxsize=AUDIO_SAMPLES)
        
        # Setup UDP socket for ADC data
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("0.0.0.0", udp_port))
            self.sock.settimeout(1)  # Non-blocking with timeout
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
    
    def clear_voltage_data(self):
        """Clear all queued voltage data"""
        try:
            while True:
                self.data_queue.get_nowait()
        except queue.Empty:
            pass
    
    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
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
                print("socket timeout")
                continue
            except Exception as e:
                print("JSON error")
                continue 
    
    def stop(self):
        self.running = False
        if hasattr(self, 'sock'):
            self.sock.close()

# -------------------------
# Video Receiving Thread
# -------------------------
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
        
        # Threading for audio classification (only if models loaded successfully)
        if audio_classifier is not None and label_encoder is not None and yamnet_model is not None:
            self.audio_classification_worker = AudioClassificationWorker(self.adc_receiver)
            self.audio_classification_worker.start()
            print("Audio classification thread started")
        else:
            self.audio_classification_worker = None
            print("Audio classification disabled - models not loaded")
        
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
        
        # Recording status label
        self.recording_status_label = tk.Label(self.video_frame, text="Recording: Ready", 
                                             font=("Arial", 10), fg="orange")
        self.recording_status_label.pack()

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

        # Voltage Graph Setup
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.fig.patch.set_facecolor('#2C3E50')
        self.ax.set_facecolor('#34495E')
        self.ax.set_ylim(-0.01, 0.01)
        self.ax.set_title("Microphone Data", color='white')
        self.ax.set_xlabel("Time", color='white')
        self.ax.set_ylabel("Magnitude", color='white')
        self.ax.tick_params(colors='white')
        self.x_data = []
        self.y_data = []
        self.time_window = 3  
        self.line, = self.ax.plot([], [], 'r-', linewidth=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Voltage update counter
        self.voltage_update_counter = 0
        self.plot_time_offset = time.time()

        # Setup VideoWriter
        self.video_out = None
        self.recording = False
        self.video_filename = None
        self.audio_filename = None
        
        # Setup Audio Recording
        self.audio_out = None
        self.audio_data_buffer = []
        self.audio_start_time = None
        self.audio_chunk_count = 0

        # Recording button
        self.record_button = tk.Button(root, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=10)

        self.update()  # start loop

    def generate_recording_filenames(self):
        """Generate timestamped filenames for video and audio recordings"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = f"recording_video_{timestamp}.mp4"
        self.audio_filename = f"recording_audio_{timestamp}.wav"
        
        print(f"Recording filenames generated:")
        print(f"  Video: {self.video_filename}")
        print(f"  Audio: {self.audio_filename}")

    def toggle_recording(self):
        if not self.recording:
            # Generate timestamped filenames
            self.generate_recording_filenames()
            
            # Start recording
            self.recording = True
            self.record_button.config(text="Stop Recording")
            self.audio_start_time = time.time()
            self.audio_data_buffer = []
            self.audio_chunk_count = 0
            
            # Initialize audio file for streaming
            self.init_audio_file()
            
            # Update status label
            self.recording_status_label.config(text=f"Recording: {self.video_filename}", fg="red")
            
            print("Recording started (video + audio)")
        else:
            # Stop recording
            self.recording = False
            self.record_button.config(text="Start Recording")
            
            # Stop video recording
            if self.video_out is not None:
                self.video_out.release()
                self.video_out = None
            
            # Finalize audio recording
            self.finalize_audio_recording()
            
            # Update status label
            self.recording_status_label.config(text="Recording: Ready", fg="orange")
            
            print("Recording stopped")
    
    def init_audio_file(self):
        """Initialize the audio file for streaming recording"""
        try:
            self.audio_out = wave.open(self.audio_filename, 'w')
            self.audio_out.setnchannels(1)  # Mono
            self.audio_out.setsampwidth(2)  # 2 bytes per sample (16-bit)
            self.audio_out.setframerate(AUDIO_SAMPLE_RATE)
            print(f"Audio file initialized: {self.audio_filename}")
        except Exception as e:
            print(f"Error initializing audio file: {e}")
            self.audio_out = None
    
    def write_audio_chunk(self):
        """Write accumulated audio data to file in chunks"""
        if self.audio_out is not None and len(self.audio_data_buffer) >= AUDIO_CHUNK_SIZE:
            try:
                # Get chunk of data
                chunk_data = self.audio_data_buffer[:AUDIO_CHUNK_SIZE]
                self.audio_data_buffer = self.audio_data_buffer[AUDIO_CHUNK_SIZE:]
                
                # Convert to audio samples
                audio_data = np.array(chunk_data, dtype=np.float32)
                
                # Normalize to [-1, 1] range
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Convert to 16-bit integers
                audio_data = (audio_data * 32767).astype(np.int16)
                
                # Write to file
                self.audio_out.writeframes(audio_data.tobytes())
                self.audio_chunk_count += 1
                
                if self.audio_chunk_count % 10 == 0:  # Print every 10 seconds
                    print(f"Audio: {self.audio_chunk_count} seconds recorded")
                    
            except Exception as e:
                print(f"Error writing audio chunk: {e}")
    
    def finalize_audio_recording(self):
        """Finalize the audio recording and close the file"""
        try:
            # Write any remaining data
            if len(self.audio_data_buffer) > 0:
                audio_data = np.array(self.audio_data_buffer, dtype=np.float32)
                
                # Normalize to [-1, 1] range
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # Convert to 16-bit integers
                audio_data = (audio_data * 32767).astype(np.int16)
                
                # Write remaining data
                if self.audio_out is not None:
                    self.audio_out.writeframes(audio_data.tobytes())
            
            # Close the file
            if self.audio_out is not None:
                self.audio_out.close()
                self.audio_out = None
                
            total_samples = (self.audio_chunk_count * AUDIO_CHUNK_SIZE) + len(self.audio_data_buffer)
            duration = total_samples / AUDIO_SAMPLE_RATE
            print(f"Audio recording completed: {self.audio_filename} ({duration:.1f} seconds)")
            
        except Exception as e:
            print(f"Error finalizing audio recording: {e}")
        finally:
            self.audio_data_buffer = []
            self.audio_chunk_count = 0
    
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
                
                # Update counters
                self.frame_counter_label.config(text=f"Frames: {self.frame_count}")
                self.calculate_fps()
                
            except Exception as e:
                print(f"Frame processing error: {e}")

        # Check for audio classification results
        #if self.audio_classification_worker is not None:
        audio_result = self.audio_classification_worker.get_result()

        if audio_result is not None:
            self.update_audio_detection_list(audio_result)
            #print(audio_result)
        


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
                        
                        # Add to audio recording buffer if recording
                        if self.recording:
                            self.audio_data_buffer.append(voltage)
                            # Write audio chunks periodically to prevent memory buildup
                            self.write_audio_chunk()
                        
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
    
    def update_audio_detection_list(self, audio_result):
        """Update the detection list with audio classification results"""
        import datetime
        
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        predicted_class = audio_result['class'].lower()
        confidence = audio_result['confidence']
        
        # Only add if this audio class hasn't been detected recently (within last 30 seconds)
        current_timestamp = time.time()
        recent_audio_detections = [d for d in self.detection_list 
                                if d.get('type') == 'audio' and 
                                (current_timestamp - d.get('timestamp', 0)) < 30]
        
        if not any(d['animal'].lower() == predicted_class for d in recent_audio_detections):
            self.detection_counter += 1
            
            # Create detection entry
            detection_frame = tk.Frame(self.detection_scrollable_frame, bg='#8E44AD', 
                                        relief=tk.RAISED, bd=1)
            detection_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Detection info
            info_text = f"#{self.detection_counter}: {predicted_class} (Audio)\nConfidence: {confidence:.2f}\nTime: {current_time}"
            detection_label = tk.Label(detection_frame, text=info_text, 
                                        bg='#8E44AD', fg='white', font=("Arial", 9),
                                        justify=tk.LEFT)
            detection_label.pack(padx=5, pady=3)
        
            # Store detection info
            self.detection_list.append({
                'frame': detection_frame,
                'animal': audio_result['class'],
                'confidence': confidence,
                'time': current_time,
                'type': 'audio',
                'timestamp': current_timestamp
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
        if hasattr(self, 'audio_classification_worker'):
            self.audio_classification_worker.stop()
        if hasattr(self, 'adc_receiver'):
            self.adc_receiver.stop()
        if hasattr(self, 'video_sock'):
            self.video_sock.close()
        if hasattr(self, 'video_receiver'):
            self.video_receiver.stop()
        if self.video_out is not None:
            self.video_out.release()
        # Save any remaining audio data
        if hasattr(self, 'recording') and self.recording:
            self.finalize_audio_recording()

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
        if hasattr(app, 'audio_classification_worker'):
            app.audio_classification_worker.stop()
        if hasattr(app, 'adc_receiver'):
            app.adc_receiver.stop()
        if hasattr(app, 'video_sock'):
            app.video_sock.close()