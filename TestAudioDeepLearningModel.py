import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import pickle
import pickle

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load trained classifier
model = load_model("yamnet_audio_classifier.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load YAMNet
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Function to extract embeddings
def extract_embedding(waveform):
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spec = yamnet_model(waveform)
    return tf.reduce_mean(embeddings, axis=0).numpy()

#------------------------------Testing using a saved audio file - Load new audio file--------------------------

# filepath = "RecordedAudio/CockatooA_1.wav"
# y, sr = librosa.load(filepath, sr=44100, mono=True)
# y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)

# # Extract embedding
# embedding = extract_embedding(y_resampled).reshape(1, -1)

# # Predict
# pred_class = le.inverse_transform([np.argmax(model.predict(embedding))])[0]
# print("Predicted class:", pred_class)


# # ----------------------Testing using the buffered audio signal values------------------------------
# Assuming audio data is stored in a list called samples
# Convert list back to numpy array
audio_np = np.array(samples, dtype=np.float32)

# Extract embedding
embedding = extract_embedding(audio_np).reshape(1, -1)

# Predict
pred_class = le.inverse_transform([np.argmax(model.predict(embedding))])[0]
print("Predicted class:", pred_class)
