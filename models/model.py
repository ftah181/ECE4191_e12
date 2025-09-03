from ultralytics import YOLO
import os

# ------------------ Configuration ------------------
def main():
    dataset_path = r"C:\Users\Josh Wright\Monash\ECE4191\models\dataset"   # path to your dataset folder
    parent_folder = os.path.dirname(dataset_path)                   # Save models and predictions one directory up
    train_images = os.path.join(dataset_path, "images/train")
    val_images = os.path.join(dataset_path, "images/val")
    test_images = os.path.join(dataset_path, "images/test")

    train_labels = os.path.join(dataset_path, "labels/train")
    val_labels = os.path.join(dataset_path, "labels/val")
    test_labels = os.path.join(dataset_path, "labels/test")

    # YAML config for YOLO
    data_yaml = os.path.join(dataset_path, "data.yaml")

    # ------------------ Train YOLOv8 ------------------
    # Choose model: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    model = YOLO("yolov8n.pt")  # Using pretrained YOLOv8n as backbone

    # Train
    model.train(
        data=data_yaml,
        epochs=250,          # adjust as needed
        batch=-1,           # auto-detect largest batch size according to available GPU VRAM
        imgsz=640,
        workers=4,
        project=os.path.join(parent_folder, "runs", "train"),
        name="my_model",
        exist_ok=True
    )

    # ------------------ Evaluate on Test Set ------------------
    metrics = model.val(data=data_yaml, split="test")
    print(metrics)

if __name__ == "__main__":
    main()