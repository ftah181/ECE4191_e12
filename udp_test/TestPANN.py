import socket
import threading
import queue
import json
import time
import numpy as np
import torch
import torch.nn as nn
import pickle
from panns_inference import AudioTagging


# =====================================================
#                USER CONFIG
# =====================================================
AUDIO_SAMPLE_RATE = 32000        
CLASSIFICATION_DURATION = 10   # seconds
SAMPLES_PER_CLASSIFICATION = int(AUDIO_SAMPLE_RATE * CLASSIFICATION_DURATION)
AUDIO_CLASSIFICATION_CONFIDENCE = 0.1

ADC_PORT = 5006
AUDIO_SAMPLES = SAMPLES_PER_CLASSIFICATION * 2


# =====================================================
#                MODEL SETUP
# =====================================================
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


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

try:
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    audio_classifier_model = torch.load("mlp_classifier_full.pth", map_location=device, weights_only=False)
    audio_classifier_model.eval()
    pann_model = AudioTagging(device=device)
    print("Audio classification models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    label_encoder = None
    audio_classifier_model = None
    pann_model = None


# =====================================================
#                AUDIO CLASSIFICATION
# =====================================================
def extract_embedding(waveform):
    """Extract PANNs CNN14 embedding from waveform"""
    if pann_model is None:
        return None
    try:
        audio_tensor = torch.tensor(waveform[None, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            _, embedding = pann_model.inference(audio_tensor)
        if not isinstance(embedding, np.ndarray):
            embedding = embedding.detach().cpu().numpy()
        return embedding
    except Exception as e:
        print(f"Embedding extraction failed: {e}")
        return None


def classify_audio(audio_data):
    """Classify a waveform using trained model + label encoder"""
    if audio_classifier_model is None or label_encoder is None:
        return None, 0.0

    try:
        audio_np = np.array(audio_data, dtype=np.float32)
        emb = extract_embedding(audio_np)
        if emb is None:
            return None, 0.0

        # Ensure embedding is 2D → average across time windows
        emb = np.array(emb)
        if emb.ndim > 1:
            emb = np.mean(emb, axis=0)

        # Now make it a tensor
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            logits = audio_classifier_model(emb_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = np.argmax(probs)
        pred_class = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        return pred_class, confidence

    except Exception as e:
        print(f"Classification error: {e}")
        return None, 0.0



# =====================================================
#                ADC RECEIVER THREAD
# =====================================================
class ADCReceiver(threading.Thread):
    def __init__(self, udp_port=ADC_PORT):
        super().__init__(daemon=True)
        self.udp_port = udp_port
        self.running = True
        self.data_queue = queue.Queue(maxsize=AUDIO_SAMPLES)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("0.0.0.0", udp_port))
            self.sock.settimeout(1)
            print(f"ADC UDP socket bound to port {udp_port}")
        except Exception as e:
            print(f"ADC socket binding error: {e}")

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
                data, _ = self.sock.recvfrom(65535)
                json_str = data.decode("utf-8")
                adc_data = json.loads(json_str)
                voltages = adc_data.get("voltages", [])
                

                if voltages:
                    
                    for v in voltages:
                        try:
                            self.data_queue.put_nowait(v)
                        except queue.Full:
                            _ = self.data_queue.get_nowait()
                            self.data_queue.put_nowait(v)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                print(f"ADCReceiver error: {e}")
                continue

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


# =====================================================
#             AUDIO CLASSIFICATION THREAD
# =====================================================
class AudioClassificationWorker(threading.Thread):
    def __init__(self, adc_receiver):
        super().__init__(daemon=True)
        self.adc_receiver = adc_receiver
        self.running = True
        self.result_queue = queue.Queue(maxsize=5)
        self.audio_buffer = []  # persistent rolling buffer
        self.last_pred_time = 0
        self.PRED_INTERVAL = 15  # run classification every 15 s max

    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            try:
                # fetch whatever is currently in the ADC queue
                new_data = self.adc_receiver.get_voltage_data()
                if new_data:
                    self.audio_buffer.extend(new_data)
                    # limit buffer size to about 25 s to avoid growth
                    max_buffer = int(AUDIO_SAMPLE_RATE * 25)
                    if len(self.audio_buffer) > max_buffer:
                        self.audio_buffer = self.audio_buffer[-max_buffer:]

                #print(f" Collected {len(self.audio_buffer)} samples in buffer...")

                # check if we have enough data (≥ CLASSIFICATION_DURATION)
                if (
                    len(self.audio_buffer) >= SAMPLES_PER_CLASSIFICATION
                    and time.time() - self.last_pred_time > self.PRED_INTERVAL
                ):
                    self.last_pred_time = time.time()

                    # take the last CLASSIFICATION_DURATION segment
                    audio_window = self.audio_buffer[-SAMPLES_PER_CLASSIFICATION:]
                    pred_class, confidence = classify_audio(audio_window)

                    if pred_class and confidence >= AUDIO_CLASSIFICATION_CONFIDENCE:
                        result = {
                            "class": pred_class,
                            "confidence": confidence,
                            "timestamp": time.time(),
                        }

                        # clear older results
                        while not self.result_queue.empty():
                            try:
                                self.result_queue.get_nowait()
                            except queue.Empty:
                                break
                        try:
                            self.result_queue.put_nowait(result)
                        except queue.Full:
                            pass

                        #print(f" Classified: {pred_class} (confidence: {confidence:.3f})")

                #time.sleep(0.2)

            except Exception as e:
                print(f"Audio classification error: {e}")
                time.sleep(1.0)



# =====================================================
#                     MAIN LOOP
# =====================================================
if __name__ == "__main__":
    adc_receiver = ADCReceiver(udp_port=ADC_PORT)
    adc_receiver.start()

    audio_worker = AudioClassificationWorker(adc_receiver)
    audio_worker.start()

    try:
        while True:
            time.sleep(1)
            result = audio_worker.get_result()
            if result:
                print(f"Latest classification result: {result}")
    except KeyboardInterrupt:
        print("\nStopping threads...")
        adc_receiver.stop()
        audio_worker.stop()
        adc_receiver.join()
        audio_worker.join()
        print("End of script\n")
