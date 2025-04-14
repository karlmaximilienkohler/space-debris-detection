from ultralytics import YOLO
import os

def train():
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize YOLO model
    # Using YOLOv8n (nano) for faster training, but you can also use:
    # - YOLOv8s (small)
    # - YOLOv8m (medium)
    # - YOLOv8l (large)∫
    # - YOLOv8x (xlarge)
    model = YOLO('yolov8n.pt')  # Load pretrained model
    
    # Train the model
    results = model.train(
        data='data/data.yaml',      # Path to data.yaml file
        epochs=100,                  # Number of epochs
        imgsz=640,                   # Image size
        batch=16,                    # Batch size
        device='cpu',                # Use CPU
        workers=0,                   # Number of workers
        patience=10,                 # Early stopping patience
        save=True,                   # Save best model
        save_period=10,              # Save every 10 epochs
        cache=False,                 # Cache images in memory
        exist_ok=True,               # Overwrite existing experiment
        pretrained=True,             # Use pretrained weights
        optimizer='auto',            # Optimizer (SGD, Adam, etc.)
        verbose=True,                # Print verbose output
        seed=42,                     # Random seed
        deterministic=True,          # Deterministic training
        rect=False,                  # Rectangular training
        cos_lr=True,                 # Cosine learning rate scheduler
        close_mosaic=10,             # Disable mosaic augmentation for last 10 epochs
        resume=False,                # Resume training from last checkpoint
        amp=True,                    # Mixed precision training
        fraction=1.0,                # Fraction of dataset to use
        label_smoothing=0.0,         # Label smoothing
        nbs=64,                      # Nominal batch size
        overlap_mask=True,           # Overlap masks
        mask_ratio=4,                # Mask downsample ratio
        dropout=0.0,                 # Dropout rate
        val=True,                    # Validate during training
        plots=True,                  # Generate plots
        save_json=False,             # Save results to JSON
        save_hybrid=False,           # Save hybrid version of labels
        conf=0.001,                  # Confidence threshold
        iou=0.6,                     # NMS IoU threshold
        max_det=300,                 # Maximum number of detections per image
        half=False,                  # Use FP16 half-precision inference
        dnn=False,                   # Use OpenCV DNN for ONNX inference
        project='runs/train',        # Project name
        name='exp'                   # Experiment name
    )
    
    # Save the final model
    model.export(format='pt')  # Export to PyTorch format
    print('Training completed!')

if __name__ == '__main__':
    train() 