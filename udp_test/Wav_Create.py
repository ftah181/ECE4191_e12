import spidev
import RPi.GPIO as GPIO
import time
import struct
import matplotlib.pyplot as plt

# -------------------- GPIO & SPI Setup --------------------
CS_PIN = 21  # GPIO21 as Chip Select
GPIO.setmode(GPIO.BCM)
GPIO.setup(CS_PIN, GPIO.OUT)
GPIO.output(CS_PIN, GPIO.HIGH)  # CS idle high

spi = spidev.SpiDev()
spi.open(0, 0)  # SPI bus 0, device 0
spi.max_speed_hz = 1350000  # 3.6 MHz

# -------------------- MCP3008 Read Function --------------------
def read_mcp3008(channel: int) -> int:
    """Read 10-bit value from MCP3008 ADC channel 0-7"""
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    return ((adc[1] & 3) << 8) | adc[2]

# -------------------- Recording Configuration --------------------
FILENAME = "AudioSample_autoRate.wav"
NUM_SAMPLES = 100000     # Total number of samples
BLOCK_SIZE = 1024        # Samples per batch write
CHANNEL = 0              # MCP3008 channel
adc_values = []
pcm_blocks = []

# -------------------- Sampling Loop --------------------
start_time = time.perf_counter()

try:
    for _ in range(NUM_SAMPLES):
        val = read_mcp3008(CHANNEL)
        adc_values.append(val)
        
        # Convert 10-bit ADC to 16-bit PCM
        pcm_val = int((val - 512) / 511 * 32767)
        pcm_val = max(min(pcm_val, 32767), -32768)
        pcm_blocks.append(struct.pack("<h", pcm_val))
        
        # Write batch to memory (optional in-memory batching)
        if len(pcm_blocks) >= BLOCK_SIZE:
            pass  # Keep in memory; write all at once later

except KeyboardInterrupt:
    print("\nRecording interrupted by user.")

end_time = time.perf_counter()
elapsed = end_time - start_time
actual_rate = len(adc_values) / elapsed
print(f"Achieved sampling rate: {actual_rate:.1f} Hz over {elapsed:.2f} seconds")

# -------------------- Write WAV File --------------------
pcm_data = b"".join(pcm_blocks)
data_bytes = len(pcm_data)

with open(FILENAME, "wb") as wf:
    # RIFF header
    wf.write(b"RIFF")
    wf.write(struct.pack("<I", 36 + data_bytes))
    wf.write(b"WAVE")
    
    # fmt subchunk
    wf.write(b"fmt ")
    wf.write(struct.pack("<IHHIIHH", 16, 1, 1, int(actual_rate), int(actual_rate*2), 2, 16))
    
    # data subchunk
    wf.write(b"data")
    wf.write(struct.pack("<I", data_bytes))
    wf.write(pcm_data)

print(f"Recording complete. WAV file saved as {FILENAME}")
print(f"Playback sample rate set to {int(actual_rate)} Hz")

# -------------------- Cleanup --------------------
spi.close()
GPIO.cleanup()

# -------------------- Plot ADC Values --------------------
plt.figure(figsize=(12, 6))
plt.plot(adc_values, color='blue')
plt.title("Raw ADC Values Over Time")
plt.xlabel("Sample Index")
plt.ylabel("ADC Value (0–1023)")
plt.grid(True)
plt.show()
