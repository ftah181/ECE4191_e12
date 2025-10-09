import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import joblib
from panns_inference import AudioTagging
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict

# --------------------
# Define same MLP model
# --------------------
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

# --------------------
# Setup
# --------------------
DATASET_PATH = "DataSetA"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model + encoder
model = torch.load("mlp_classifier_full.pth", map_location=device, weights_only=False)
model.eval()

le = joblib.load("label_encoder.pkl")
classes = list(le.classes_)

# Load PANNs
at = AudioTagging(device=device)

# --------------------
# Embedding extraction
# --------------------
def extract_embedding(audio, sr=44100, target_sr=32000):
    try:
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        audio_tensor = torch.tensor(audio[None, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            _, embedding = at.inference(audio_tensor)

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy().squeeze()
        else:
            embedding = np.array(embedding).squeeze()
        return embedding
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

# --------------------
# Prediction helper
# --------------------
def predict(audio_path):
    """Return predicted class name for one .wav file"""
    audio, sr = librosa.load(audio_path, sr=44100, mono=True)
    emb = extract_embedding(audio, sr=sr, target_sr=32000)
    if emb is None:
        return None
    emb = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(emb)
        pred = torch.argmax(logits, dim=1).item()
    return le.inverse_transform([pred])[0]



# --------------------
# Loop through dataset -testing
# --------------------
print("\n🔍 Evaluating all audio files in:", DATASET_PATH)
y_true, y_pred = [], []

for fname in os.listdir(DATASET_PATH):
    if not fname.lower().endswith(".wav"):
        continue

    # True label = prefix before first underscore or match against known class names
    true_label = None
    for c in sorted(classes, key=lambda x: -len(x)):
        if fname.startswith(c):
            true_label = c
            break
    if true_label is None:
        true_label = "Null"

    fpath = os.path.join(DATASET_PATH, fname)
    predicted = predict(fpath)

    y_true.append(true_label)
    y_pred.append(predicted if predicted is not None else "Error")

    print(f"🎧 {fname:40s}  True: {true_label:12s}  Pred: {predicted}")

# --------------------
# Evaluation metrics
# --------------------
print("\n📊 Overall Results")
acc = accuracy_score(y_true, y_pred)
print(f"✅ Overall Accuracy: {acc * 100:.2f}%\n")

print("📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
