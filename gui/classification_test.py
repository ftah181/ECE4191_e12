#!/usr/bin/env python3
"""
Test Animal Detection on Static Images
"""

import cv2
import os
from inference_sdk import InferenceHTTPClient
import json

def test_detection_on_image(image_path):
    """Test animal detection on a single image"""
    
    # Setup Roboflow client
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="lnHqcMh4NynT1If5FC38"  # Replace with your actual key
    )
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    print(f"Processing image: {image_path}")
    
    try:
        # Run inference
        print("Running inference...")
        result = client.infer(image_path, model_id="animal-detection-evlon/1")
        
        # Load image for display
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image!")
            return
        
        # Get predictions
        predictions = result.get("predictions", [])
        print(f"\nDetection Results:")
        print(f"Number of animals detected: {len(predictions)}")
        
        # Process each prediction
        for i, pred in enumerate(predictions, 1):
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            class_name = pred["class"]
            confidence = pred["confidence"]
            
            print(f"\n{i}. {class_name}")
            print(f"   Confidence: {confidence:.2f} ({confidence*100:.1f}%)")
            print(f"   Position: x={x:.0f}, y={y:.0f}")
            print(f"   Size: {w:.0f} x {h:.0f}")
            
            # Convert x, y, w, h to top-left and bottom-right points
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with background
            label = f"{class_name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 0, 0), 2)
        
        # Save result image
        output_path = f"detected_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, image)
        print(f"\nResult image saved as: {output_path}")
        
        # Save detection data as JSON
        json_output = f"detection_results_{os.path.splitext(os.path.basename(image_path))[0]}.json"
        with open(json_output, 'w') as f:
            json.dump({
                'image_path': image_path,
                'predictions': predictions,
                'total_detections': len(predictions)
            }, f, indent=2)
        print(f"Detection data saved as: {json_output}")
        
        # Display image (optional - comment out if running headless)
        try:
            cv2.imshow("Detection Result", image)
            print("\nPress any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Could not display image (headless mode)")
            
    except Exception as e:
        print(f"Error during detection: {e}")

def test_multiple_images(folder_path):
    """Test detection on all images in a folder"""
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
        return
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in '{folder_path}'")
        return
    
    print(f"Found {len(image_files)} image(s) in '{folder_path}'")
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print("\n" + "="*50)
        test_detection_on_image(image_path)

if __name__ == "__main__":
    print("Animal Detection Test Script")
    print("="*40)
    
    # Test options
    print("\nChoose an option:")
    print("1. Test single image")
    print("2. Test all images in a folder")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        image_path = input("Enter image path: ").strip()
        test_detection_on_image(image_path)
        
    elif choice == "2":
        folder_path = input("Enter folder path: ").strip()
        test_multiple_images(folder_path)
        
    else:
        # Default test - modify these paths for your images
        print("\nUsing default test...")
        
        # Try common image names/locations
        test_images = [
            "test.jpg",
            "image.jpg", 
            "animal.jpg",
            "High light/test.jpg",  # Based on your original script
        ]
        
        found_image = False
        for test_path in test_images:
            if os.path.exists(test_path):
                print(f"Testing with: {test_path}")
                test_detection_on_image(test_path)
                found_image = True
                break
        
        if not found_image:
            print("No test images found. Please specify an image path manually.")
            image_path = input("Enter image path: ").strip()
            if image_path:
                test_detection_on_image(image_path)