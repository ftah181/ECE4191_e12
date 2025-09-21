import cv2
from ultralytics import YOLO
from utils.config import MODEL_PATH, CONF_THRESHOLD
from utils.helpers import draw_predictions

class YOLOModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model = YOLO(model_path)
        self.last_predictions = []
        self.frame_skip_counter = 0
    
    def predict(self, frame, force_inference=False, skip_frames=30):
        """Run inference on frame with frame skipping optimization"""
        self.frame_skip_counter += 1
        if not force_inference and self.frame_skip_counter < skip_frames:
            return draw_predictions(frame, self.last_predictions)

        self.frame_skip_counter = 0
        predictions = []

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame_rgb, 
                                imgsz=640,
                                conf=CONF_THRESHOLD,
                                iou=0.45,
                                max_det=300,
                                augment=False,
                                agnostic_nms=False)

            res = results[0]
            predictions = []

            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
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

            self.last_predictions = predictions

        except Exception as e:
            print(f"Inference error: {e}")
            predictions = self.last_predictions

        return draw_predictions(frame, predictions), predictions