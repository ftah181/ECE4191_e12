from ultralytics import YOLO

# Load pretrained model
model = YOLO("models/runs/train/my_model/weights/best.pt")

# Inference
results = model("img.jpg")   # run detection on an image

# Access the first result
res = results[0]

# Show the detections
res.show()
