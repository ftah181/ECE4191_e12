from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()

config = picam2.create_still_configuration(
    main={
    "size": (3280, 2464),
    })

picam2.configure(config)

controls = {
    "AwbMode": 4,
    "AeEnable": True,                  # enable auto exposure
    "ColourGains": (1.2, 1.1),  # boost red slightly, blue slightly
    "AnalogueGain": 1.0                # default gain
}
picam2.set_controls(controls)

picam2.start()
time.sleep(2)

picam2.capture_file("test_photo.jpg")
#print(picam2.camera_controls)


picam2.stop()
