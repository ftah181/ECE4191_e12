import os
import cv2
from inference_sdk import InferenceHTTPClient

# --- Setup Roboflow client ---
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="lnHqcMh4NynT1If5FC38"  # Replace with your actual key
)

# --- Path to your images ---
image_folder = "High light"

# --- Loop through each image ---
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        
        # Run inference
        result = client.infer(image_path, model_id="animal-detection-evlon/1")
        
        # Load image for drawing
        image = cv2.imread(image_path)

        # Draw bounding boxes from results
        for pred in result.get("predictions", []):
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            class_name = pred["class"]
            confidence = pred["confidence"]

            # Convert x, y, w, h to top-left and bottom-right points
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # Show image
        cv2.imshow("Result", image)
        cv2.waitKey(0)  # Wait until key press before showing next image

cv2.destroyAllWindows()
