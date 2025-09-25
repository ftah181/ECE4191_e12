import pyaudio
import wave
import time

# -------------------------
# Settings
# -------------------------
CHUNK = 1024             # Number of samples per frame
FORMAT = pyaudio.paInt16 # 16-bit resolution
CHANNELS = 1             # Mono audio
RATE = 16000             # 16 kHz sample rate (for YAMNet compatibility)
RECORD_SECONDS = 30       # Duration to record
OUTPUT_FILENAME = "Motor_Audio.wav"

# -------------------------
# Recording
# -------------------------
p = pyaudio.PyAudio()

print("Available audio devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"{i}: {info['name']} (Max input channels: {info['maxInputChannels']})")

# If needed, you can manually set device_index below
DEVICE_INDEX = None  # or replace with an integer from the printed list

print("\nRecording started...")

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=CHUNK)

frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

print("Recording finished.")

# -------------------------
# Save to WAV
# -------------------------
stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"Saved recording as {OUTPUT_FILENAME}")
