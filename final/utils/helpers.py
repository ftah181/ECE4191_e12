import cv2
import numpy as np

def resize_for_display(frame, max_width=800, max_height=600, min_width=320, min_height=240):
    """Resize frame for display with reasonable size limits"""
    height, width = frame.shape[:2]
    
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

def draw_predictions(frame, predictions, conf_threshold=0.7):
    """Draw bounding boxes and labels on frame"""
    try:
        color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        for pred in predictions:
            if pred["confidence"] > conf_threshold:
                x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Label
                label = f"{pred['class'][:8]} {pred['confidence']:.1f}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            font, font_scale, color, thickness)

    except Exception as e:
        print(f"Draw error: {e}")

    return frame