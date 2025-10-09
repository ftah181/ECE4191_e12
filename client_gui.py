#!/usr/bin/env python3
import pygame
import cv2
import numpy as np
import socket
import threading
import queue
import time
import json
from ultralytics import YOLO
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime
import wave
from panns_inference import AudioTagging
import torch
import torch.nn as nn

# Configuration
WIDTH, HEIGHT = 1400, 800
VIDEO_WIDTH, VIDEO_HEIGHT = 800, 600
GRAPH_WIDTH, GRAPH_HEIGHT = 800, 200
INFO_PANEL_WIDTH = 600

# Colors
BG_COLOR = (44, 62, 80)
PANEL_COLOR = (52, 73, 94)
TEXT_COLOR = (255, 255, 255)
BUTTON_COLOR = (231, 76, 60)
DETECTION_BG = (44, 62, 80)
AUDIO_DETECTION_BG = (142, 68, 173)
LINE_COLOR = (30, 30, 30)

# Model paths and parameters
MODEL_PATH = "Model_HL_16-09.pt"
AUDIO_MODEL_PATH = "./yamnet_model"
LABEL_ENCODER_PATH = "label_encoder.pkl"
AUDIO_CLASSIFIER_PATH = "yamnet_audio_classifier.h5"

INFERENCE_SKIP_FRAMES = 30
CONF_THRESHOLD = 0.7
AUDIO_CLASSIFICATION_INTERVAL = 20
AUDIO_CLASSIFICATION_CONFIDENCE = 0.7
YAMNET_SAMPLE_RATE = 32000
AUDIO_SAMPLE_RATE = 32000
AUDIO_CHUNK_SIZE = 32000
AUDIO_SAMPLES = 1024 * 200

# UDP ports
VIDEO_PORT = 5005
ADC_PORT = 5006

# Global variables
frame_skip_counter = 0
last_predictions = []

# Load models
try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# ----------------------------------------
#               OLD MODEL
# ---------------------------------------
# try:
#     with open(LABEL_ENCODER_PATH, "rb") as f:
#         label_encoder = pickle.load(f)
#     audio_classifier = load_model(AUDIO_CLASSIFIER_PATH)
#     yamnet_model = hub.load(AUDIO_MODEL_PATH)
#     print("Audio classification models loaded successfully")
# except Exception as e:
#     print(f"Error loading audio models: {e}")
#     label_encoder = None
#     audio_classifier = None
#     yamnet_model = None

# ----------------------------------------
#               NEW MODEL
# ---------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_classes=15):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.model(x)
    
# Load audio classification models
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

try:
    # Load label encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Load trained classifier (PyTorch)
    audio_classifier_model = torch.load("mlp_classifier_full.pth", map_location=device, weights_only=False)
    audio_classifier_model.eval()

    # Load pretrained PANNs feature extractor
    pann_model = AudioTagging(device=device)

    print("✅ Audio classification models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    label_encoder = None
    audio_classifier_model = None
    pann_model = None

# OLD: Audio classification functions
# def extract_embedding(waveform):
#     if yamnet_model is None:
#         return None
#     try:
#         waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
#         scores, embeddings, spec = yamnet_model(waveform)
#         return tf.reduce_mean(embeddings, axis=0).numpy()
#     except Exception as e:
#         print(f"Error extracting embedding: {e}")
#         return None


# def classify_audio(audio_data):
#     if audio_classifier is None or label_encoder is None:
#         return None, 0.0
#     try:
#         if isinstance(audio_data, list):
#             audio_np = np.array(audio_data, dtype=np.float32)
#         else:
#             audio_np = audio_data.astype(np.float32)
        
#         if len(audio_np) > 0 and len(audio_np) != YAMNET_SAMPLE_RATE:
#             indices = np.linspace(0, len(audio_np) - 1, YAMNET_SAMPLE_RATE)
#             audio_np = np.interp(indices, np.arange(len(audio_np)), audio_np)
        
#         embedding = extract_embedding(audio_np)
#         if embedding is None:
#             return None, 0.0
        
#         embedding = embedding.reshape(1, -1)
#         prediction = audio_classifier.predict(embedding, verbose=0)
#         predicted_class_idx = np.argmax(prediction)
#         confidence = float(prediction[0][predicted_class_idx])
#         predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        
#         return predicted_class, confidence
#     except Exception as e:
#         print(f"Error classifying audio: {e}")
#         return None, 0.0

def extract_embedding(waveform):
    """Extract PANNs CNN14 embedding"""
    if pann_model is None:
        return None
    
    try:
        audio_tensor = torch.tensor(waveform[None, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            _, embedding = pann_model.inference(audio_tensor)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy().squeeze()
        else:
            embedding = np.array(embedding).squeeze()
        return embedding
    except Exception as e:
        print(f"Embedding extraction failed: {e}")
        return None

def classify_audio(audio_data):
    """Classify audio data using the trained model"""
    if audio_classifier_model is None or label_encoder is None:
        return None, 0.0
    
    try:
        # Convert list of dicts or floats to numpy array
        if isinstance(audio_data[0], dict) and "voltage" in audio_data[0]:
            audio_np = np.array([v["voltage"] for v in audio_data], dtype=np.float32)
        else:
            audio_np = np.array(audio_data, dtype=np.float32)

        # Extract embedding
        emb = extract_embedding(audio_np)
        if emb is None:
            return None, 0.0

        # Classify
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = audio_classifier_model(emb_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = np.argmax(probs)
        pred_class = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])

        return pred_class, confidence

    except Exception as e:
        print(f"⚠️ Classification error: {e}")
        return None, 0.0
    

# Video inference
def model_predict(frame):
    global frame_skip_counter, last_predictions
    
    frame_skip_counter += 1
    if frame_skip_counter < INFERENCE_SKIP_FRAMES:
        return draw_predictions(frame, last_predictions)
    
    frame_skip_counter = 0
    predictions = []
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, imgsz=640, conf=0.25, iou=0.45, max_det=300, augment=False, agnostic_nms=False)
        res = results[0]
        
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].item())
            cls = int(box.cls[0].item())
            class_name = res.names[cls]
            
            predictions.append({
                "class": class_name,
                "confidence": confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
        
        last_predictions = predictions
    except Exception as e:
        print(f"Inference error: {e}")
        predictions = last_predictions
    
    return draw_predictions(frame, predictions)


def draw_predictions(frame, predictions):
    for pred in predictions:
        if pred["confidence"] > CONF_THRESHOLD:
            x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{pred['class'][:8]} {pred['confidence']:.1f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame, predictions


# Threading classes
class VideoReceiver(threading.Thread):
    def __init__(self, udp_ip, video_port):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576 * 4)
        self.sock.bind((udp_ip, video_port))
        self.sock.setblocking(False)
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
    
    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1048576)
                npdata = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
                if frame is not None:
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


class ADCReceiver(threading.Thread):
    def __init__(self, udp_port=5006):
        super().__init__(daemon=True)
        self.udp_port = udp_port
        self.running = True
        self.latest_voltage = 0.0
        self.data_queue = queue.Queue(maxsize=AUDIO_SAMPLES)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("0.0.0.0", udp_port))
            self.sock.settimeout(1)
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
    
    def clear_voltage_data(self):
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
                voltages = adc_data.get("voltages", [])
                timestamp = adc_data.get("timestamp", time.time())
                
                if voltages:
                    self.latest_voltage = voltages[-1]
                    for v in voltages:
                        entry = {"voltage": v, "timestamp": timestamp}
                        try:
                            self.data_queue.put_nowait(entry)
                        except queue.Full:
                            try:
                                self.data_queue.get_nowait()
                                self.data_queue.put_nowait(entry)
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


class AudioClassificationWorker(threading.Thread):
    def __init__(self, adc_receiver):
        super().__init__(daemon=True)
        self.adc_receiver = adc_receiver
        self.running = True
        self.last_classification_time = 0
        self.result_queue = queue.Queue(maxsize=5)
    
    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def run(self):
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_classification_time >= AUDIO_CLASSIFICATION_INTERVAL:
                    audio_data = self.adc_receiver.get_voltage_data()
                    if len(audio_data) > 0:
                        voltage_values = [entry['voltage'] for entry in audio_data]
                        predicted_class, confidence = classify_audio(voltage_values)
                        
                        if predicted_class is not None and confidence >= AUDIO_CLASSIFICATION_CONFIDENCE:
                            result = {
                                'class': predicted_class,
                                'confidence': confidence,
                                'timestamp': current_time
                            }
                            
                            try:
                                while not self.result_queue.empty():
                                    self.result_queue.get_nowait()
                            except queue.Empty:
                                pass
                            
                            try:
                                self.result_queue.put_nowait(result)
                            except queue.Full:
                                pass
                            
                            print(f"Audio classification: {predicted_class} (confidence: {confidence:.3f})")
                    
                    self.adc_receiver.clear_voltage_data()
                    self.last_classification_time = current_time
                
                time.sleep(0.1)
            except Exception as e:
                print(f"Audio classification error: {e}")
                time.sleep(1.0)
    
    def stop(self):
        self.running = False


# Main GUI Class
class PygameGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Audio/Visual Classification GUI")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Threads
        self.video_receiver = VideoReceiver("0.0.0.0", VIDEO_PORT)
        self.adc_receiver = ADCReceiver(ADC_PORT)
        self.audio_worker = AudioClassificationWorker(self.adc_receiver) if audio_classifier_model else None
        
        self.video_receiver.start()
        self.adc_receiver.start()
        if self.audio_worker:
            self.audio_worker.start()
        
        # State
        self.latest_frame = None
        self.processed_frame = None
        self.detections = []
        self.detected_animals = set()
        self.detection_counter = 0
        self.audio_data = []
        self.fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.recording = False
        self.video_out = None
        self.audio_out = None
        self.audio_buffer = []
        self.video_filename = None
        self.audio_filename = None
        
        # Scrolling for detections
        self.scroll_offset = 0
        self.max_scroll = 0
    
    def toggle_recording(self):
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = f"recording_video_{timestamp}.mp4"
            self.audio_filename = f"recording_audio_{timestamp}.wav"
            self.recording = True
            self.audio_buffer = []
            
            # Initialize audio file
            try:
                self.audio_out = wave.open(self.audio_filename, 'w')
                self.audio_out.setnchannels(1)
                self.audio_out.setsampwidth(2)
                self.audio_out.setframerate(AUDIO_SAMPLE_RATE)
                print(f"Recording started: {self.video_filename}, {self.audio_filename}")
            except Exception as e:
                print(f"Error initializing audio file: {e}")
                self.audio_out = None
        else:
            self.recording = False
            if self.video_out:
                self.video_out.release()
                self.video_out = None
            if self.audio_out:
                if len(self.audio_buffer) > 0:
                    audio_data = np.array(self.audio_buffer, dtype=np.float32)
                    if np.max(np.abs(audio_data)) > 0:
                        audio_data = audio_data / np.max(np.abs(audio_data))
                    audio_data = (audio_data * 32767).astype(np.int16)
                    self.audio_out.writeframes(audio_data.tobytes())
                self.audio_out.close()
                self.audio_out = None
            print("Recording stopped")
    
    def add_detection(self, class_name, confidence, is_audio=False):
        current_time = datetime.now().strftime("%H:%M:%S")
        animal_key = class_name.lower()
        
        if not is_audio:
            if animal_key not in self.detected_animals:
                self.detected_animals.add(animal_key)
                self.detection_counter += 1
                self.detections.append({
                    'id': self.detection_counter,
                    'class': class_name,
                    'confidence': confidence,
                    'time': current_time,
                    'is_audio': False
                })
        else:
            # For audio, add regardless (timestamp-based filtering handled in worker)
            self.detection_counter += 1
            self.detections.append({
                'id': self.detection_counter,
                'class': class_name,
                'confidence': confidence,
                'time': current_time,
                'is_audio': True
            })
    
    def clear_detections(self):
        self.detections.clear()
        self.detected_animals.clear()
        self.detection_counter = 0
        self.scroll_offset = 0
    
    def draw_video(self):
        # Video display area
        video_rect = pygame.Rect(10, 10, VIDEO_WIDTH, VIDEO_HEIGHT)
        pygame.draw.rect(self.screen, (0, 0, 0), video_rect)
        
        if self.processed_frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.rot90(frame_rgb)
            frame_rgb = np.flipud(frame_rgb)
            
            # Resize to fit display
            h, w = self.processed_frame.shape[:2]
            scale = min(VIDEO_WIDTH / w, VIDEO_HEIGHT / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB), (new_w, new_h))
            
            surf = pygame.surfarray.make_surface(np.rot90(np.flipud(frame_resized)))
            x_offset = (VIDEO_WIDTH - new_w) // 2
            y_offset = (VIDEO_HEIGHT - new_h) // 2
            self.screen.blit(surf, (10 + x_offset, 10 + y_offset))
    
    def draw_graph(self):
        # Audio waveform graph
        graph_rect = pygame.Rect(10, VIDEO_HEIGHT + 20, VIDEO_WIDTH, GRAPH_HEIGHT)
        pygame.draw.rect(self.screen, (20, 20, 20), graph_rect)
        
        if len(self.audio_data) > 1:
            points = []
            for i, voltage in enumerate(self.audio_data[-2000:]):  # Last 2000 samples
                x = 10 + (i / 2000) * VIDEO_WIDTH
                y = VIDEO_HEIGHT + 20 + GRAPH_HEIGHT // 2 - (voltage * GRAPH_HEIGHT // 2)
                points.append((int(x), int(y)))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 200, 0), False, points, 1)
    
    def draw_info_panel(self):
        panel_x = VIDEO_WIDTH + 20
        panel_rect = pygame.Rect(panel_x, 10, INFO_PANEL_WIDTH, HEIGHT - 20)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel_rect)
        
        # Title
        title = self.font_large.render("Detected Animals", True, TEXT_COLOR)
        self.screen.blit(title, (panel_x + 10, 20))
        
        # Clear button
        button_rect = pygame.Rect(panel_x + INFO_PANEL_WIDTH - 120, 20, 100, 40)
        pygame.draw.rect(self.screen, BUTTON_COLOR, button_rect)
        button_text = self.font_small.render("Clear List", True, TEXT_COLOR)
        self.screen.blit(button_text, (button_rect.x + 10, button_rect.y + 12))
        
        # Record button
        record_button_rect = pygame.Rect(panel_x + 10, HEIGHT - 70, 200, 50)
        record_color = (200, 50, 50) if self.recording else (50, 150, 50)
        pygame.draw.rect(self.screen, record_color, record_button_rect)
        record_text = self.font_medium.render("Stop Recording" if self.recording else "Start Recording", True, TEXT_COLOR)
        self.screen.blit(record_text, (record_button_rect.x + 10, record_button_rect.y + 15))
        
        # Detection list
        list_y = 80
        list_height = HEIGHT - 150
        detection_area = pygame.Rect(panel_x + 10, list_y, INFO_PANEL_WIDTH - 20, list_height)
        pygame.draw.rect(self.screen, DETECTION_BG, detection_area)
        
        # Draw detections with scrolling
        y_offset = list_y + 10 - self.scroll_offset
        item_height = 60
        
        for det in self.detections:
            if y_offset > list_y - item_height and y_offset < list_y + list_height:
                det_rect = pygame.Rect(panel_x + 15, y_offset, INFO_PANEL_WIDTH - 30, item_height - 5)
                bg_color = AUDIO_DETECTION_BG if det['is_audio'] else DETECTION_BG
                pygame.draw.rect(self.screen, bg_color, det_rect)
                pygame.draw.rect(self.screen, (100, 100, 100), det_rect, 1)
                
                # Text
                id_text = self.font_small.render(f"#{det['id']}: {det['class']}", True, TEXT_COLOR)
                conf_text = self.font_small.render(f"Confidence: {det['confidence']:.2f}", True, TEXT_COLOR)
                time_text = self.font_small.render(f"Time: {det['time']}", True, TEXT_COLOR)
                
                self.screen.blit(id_text, (det_rect.x + 5, det_rect.y + 5))
                self.screen.blit(conf_text, (det_rect.x + 5, det_rect.y + 23))
                self.screen.blit(time_text, (det_rect.x + 5, det_rect.y + 41))
            
            y_offset += item_height
        
        # Update max scroll
        self.max_scroll = max(0, len(self.detections) * item_height - list_height)
        
        # FPS and status
        status_y = list_y + list_height + 10
        fps_text = self.font_small.render(f"FPS: {self.fps:.1f}", True, TEXT_COLOR)
        self.screen.blit(fps_text, (panel_x + 10, status_y))
        
        return button_rect, record_button_rect, detection_area
    
    def update(self):
        # Get new frame
        new_frame = self.video_receiver.get_latest_frame()
        if new_frame is not None:
            self.latest_frame = new_frame
            
            # Run inference
            if model is not None:
                self.processed_frame, predictions = model_predict(new_frame.copy())
                
                # Add new detections
                for pred in predictions:
                    if pred['confidence'] > CONF_THRESHOLD:
                        self.add_detection(pred['class'], pred['confidence'], is_audio=False)
            else:
                self.processed_frame = new_frame
            
            # Recording
            if self.recording:
                if self.video_out is None:
                    h, w = new_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_out = cv2.VideoWriter(self.video_filename, fourcc, 20.0, (w, h))
                
                if self.video_out:
                    self.video_out.write(new_frame)
            
            # FPS calculation
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
        
        # Get audio data
        voltage_data = self.adc_receiver.get_voltage_data()
        for entry in voltage_data:
            self.audio_data.append(entry['voltage'])
            if self.recording and self.audio_out:
                self.audio_buffer.append(entry['voltage'])
        
        # Keep audio data limited
        if len(self.audio_data) > 10000:
            self.audio_data = self.audio_data[-10000:]
        
        # Check for audio classifications
        if self.audio_worker:
            audio_result = self.audio_worker.get_result()
            if audio_result:
                self.add_detection(audio_result['class'], audio_result['confidence'], is_audio=True)
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    
                    # Check button clicks
                    button_rect, record_button_rect, detection_area = self.draw_info_panel()
                    
                    if button_rect.collidepoint(mouse_pos):
                        self.clear_detections()
                    elif record_button_rect.collidepoint(mouse_pos):
                        self.toggle_recording()
                    
                    # Scroll handling
                    if event.button == 4:  # Scroll up
                        self.scroll_offset = max(0, self.scroll_offset - 30)
                    elif event.button == 5:  # Scroll down
                        self.scroll_offset = min(self.max_scroll, self.scroll_offset + 30)
            
            # Update state
            self.update()
            
            # Draw everything
            self.screen.fill(BG_COLOR)
            self.draw_video()
            self.draw_graph()
            self.draw_info_panel()
            
            pygame.display.flip()
            self.clock.tick(30)
        
        # Cleanup
        self.video_receiver.stop()
        self.adc_receiver.stop()
        if self.audio_worker:
            self.audio_worker.stop()
        if self.video_out:
            self.video_out.release()
        if self.audio_out:
            self.audio_out.close()
        pygame.quit()


if __name__ == "__main__":
    gui = PygameGUI()
    gui.run()