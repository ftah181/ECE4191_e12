from ultralytics import YOLO
import os
from datetime import datetime

def main():
    # Path to your trained model
    model_path = r"C:\Users\Josh Wright\Monash\ECE4191\models\runs\train\my_model\weights\best.pt"
    model = YOLO(model_path)

    # Path to your new test images
    rpi_folder = r"C:\Users\Josh Wright\Monash\ECE4191\models\dataset\images\RPi_photos"

    # Path to save location
    project_folder = r"C:\Users\Josh Wright\Monash\ECE4191\models\runs\predict"

    # Generate a name with the current date
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g., '20250902_1830'
    run_name = f"RPi_inference_{date_str}"

    # Run predictions
    results = model.predict(
        source=rpi_folder,          # folder with new images
        imgsz=640,
        conf=0.6,
        save=True,                  # save annotated images
        project=project_folder,  # where to save predictions
        name=run_name,       # folder name for this inference run
    )

if __name__ == "__main__":
    main()