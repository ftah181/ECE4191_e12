from roboflow import Roboflow
from inference import get_model
import cv2

# Load model
model = get_model("animal-detection-evlon/3", api_key="lnHqcMh4NynT1If5FC38")

# Run inference
results = model.infer("test.jpg")

# Debug: Print results structure
print("Results type:", type(results))
print("Results content:", results)
print("Results attributes:", dir(results) if hasattr(results, '__dict__') else "No attributes")

# Load the image
image = cv2.imread("test.jpg")
if image is None:
    print("Error: Could not load image 'test.jpg'")
    exit()

predictions_found = 0

# Process results - Handle list of ObjectDetectionInferenceResponse objects
print("Processing results...")

# Results is a list of ObjectDetectionInferenceResponse objects
for response_idx, response in enumerate(results):
    print(f"Processing response {response_idx}")
    print(f"Response has {len(response.predictions)} predictions")
    
    # Process each prediction in this response
    for pred_idx, prediction in enumerate(response.predictions):
        print(f"Processing prediction {pred_idx}: {prediction}")
        
        # Get coordinates and info from the prediction object
        x = prediction.x
        y = prediction.y
        width = prediction.width
        height = prediction.height
        confidence = prediction.confidence
        class_name = prediction.class_name
        
        print(f"Drawing: {class_name} at ({x}, {y}) size ({width}x{height}) conf: {confidence}")
        
        # Convert center coordinates to corner coordinates
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)
        
        print(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw label with background for better visibility
        label = f"{class_name} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(image, (x1, y1-text_height-10), (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(image, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        predictions_found += 1
        print(f"Successfully drew: {class_name} with confidence {confidence:.3f}")

print(f"Finished processing. Drew {predictions_found} predictions.")

print(f"Total predictions: {predictions_found}")

# Save result
cv2.imwrite("result_with_predictions.jpg", image)
print("Result saved as 'result_with_predictions.jpg'")

# Display result (optional - comment out if no display available)
try:
    cv2.imshow("Predictions", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except cv2.error:
    print("Display not available, but image saved successfully")