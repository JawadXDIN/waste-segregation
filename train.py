import os
from ultralytics import YOLO

def download_dataset():
    """
    Instructions for obtaining the dataset:
    
    1. Dataset on Kaggle: 
       Search for "Waste Classification Dataset" or "Garbage Classification" on Kaggle.
       Example: https://www.kaggle.com/datasets/techsash/waste-classification-data
       
    2. To add custom objects like "Pen", "Pencil", "Plastic Bottle", "Cardboard":
       - It is highly recommended to use Roboflow (https://roboflow.com/)
       - You can upload a short video of your pen, pencil, cardboard, and bottle.
       - Roboflow will automatically split it into images.
       - Draw boxes around the pen, pencil, etc., and export the dataset in "YOLOv11" format.
       - Extract the downloaded zip file into this 'dataset' folder.
       - Make sure your 'dataset.yaml' matches the classes you created.
       
    Make sure your dataset is structured according to the YOLO format,
    and update dataset/dataset.yaml accordingly.
    """
    pass

def train_model():
    print("Initializing YOLOv11 model for training...")
    
    # Load a pre-trained YOLOv11 nano model
    model = YOLO("yolo11n.pt") 
    
    # Ensure the dataset path exists
    yaml_path = os.path.abspath("dataset/dataset.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        print("Please ensure your dataset is placed in the 'dataset' folder.")
        print("Refer to the instructions in train.py to download datasets.")
        return

    # Train the model
    print(f"Training on dataset: {yaml_path}")
    
    results = model.train(
        data=yaml_path,
        epochs=50,       # Adjust based on dataset size and time available
        imgsz=416,       # Input image size
        batch=8,         # Batch size
        workers=2,
        project="runs",
        name="waste_detection",
        exist_ok=True    # Overwrite if exists
    )
    
    print("\nTraining Complete!")
    print("Best model weights saved to: runs/waste_detection/weights/best.pt")
    print("Copy this file to the 'model' directory to use it in the web app.")

if __name__ == "__main__":
    download_dataset()
    train_model()
