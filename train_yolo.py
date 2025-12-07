from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Check if GPU is available
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: GPU not detected! Training will be very slow.")

    # Loading YOLOv8n model (nano - smallest and fastest)
    print("\nResuming training from last checkpoint...")  # Use this if you want from start -> print("\nLoading YOLOv8n model...")
    model = YOLO('runs/detect/train/weights/last.pt')     # model = YOLO('yolov8n.pt') As we had to resume soo this is used

    # Training the model
    print("\nStarting training...")
    print("This will continue from epoch 68.")  # Use this if starting from scratch -> print("This will take several hours depending on your dataset size.")
    print("You can monitor progress in the terminal and view results in runs/detect/train/\n")

    results = model.train(
    data='D:\\military_object_dataset\\military_dataset.yaml',
    epochs=100,           # Number of training cycles through the dataset
    imgsz=640,            # Image size (640x640 pixels)
    batch=16,             # For 4Gb GPU using 16 soo ~3GB GPU memory safely used.
    device=0,             # GPU device ID (0 = first GPU, 'cpu' for CPU)
    workers=2,            # Worker threads for data loading
    patience=20,          # Stop if no improvement for 20 epochs
    save=True,            # Save checkpoints
    project='runs/detect', # Where to save results
    name='train',         # Experiment name
    exist_ok=True,        # Overwrite existing experiment
    resume=True,          # Resume training from last checkpoint
    pretrained=True,      # Use pretrained weights (recommended)
    optimizer='SGD',      # Optimizer type
    verbose=True,         # Print detailed logs
    seed=42,              # Random seed for reproducibility
    deterministic=False,  # Non-deterministic for faster training
    single_cls=False,     # False for multi-class detection
    rect=False,           # Rectangular training (can reduce padding)
    cos_lr=True,          # Cosine learning rate scheduler
    close_mosaic=10,      # Disable mosaic augmentation for last N epochs
    amp=True,             # Automatic Mixed Precision enabled for speed
    fraction=1.0,         # Use 100% of training data
    profile=False,        # Profile ONNX and TensorRT speeds
    freeze=None,          # Freeze first N layers (None = don't freeze)
    lr0=0.01,             # Initial learning rate
    lrf=0.01,             # Final learning rate (as fraction of lr0)
    momentum=0.937,       # SGD momentum
    weight_decay=0.0005,  # Weight decay (L2 regularization)
    warmup_epochs=3.0,    # Warmup epochs
    warmup_momentum=0.8,  # Warmup momentum
    box=7.5,              # Box loss weight
    cls=0.5,              # Class loss weight
    dfl=1.5,              # DFL loss weight
    hsv_h=0.015,          # HSV-Hue augmentation
    hsv_s=0.7,            # HSV-Saturation augmentation
    hsv_v=0.4,            # HSV-Value augmentation
    degrees=0.0,          # Rotation augmentation (degrees)
    translate=0.1,        # Translation augmentation
    scale=0.5,            # Scaling augmentation
    shear=0.0,            # Shear augmentation (degrees)
    perspective=0.0,      # Perspective augmentation
    flipud=0.0,           # Vertical flip probability
    fliplr=0.5,           # Horizontal flip probability
    mosaic=1.0,           # Mosaic augmentation probability
    mixup=0.0,            # MixUp augmentation probability
    copy_paste=0.0        # Copy-paste augmentation probability
    )

    # Print training summary
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"\nBest model saved to: runs/detect/train/weights/best.pt")
    print(f"Last model saved to: runs/detect/train/weights/last.pt")
    print(f"\nTraining results:")
    print(f"  - Charts and metrics: runs/detect/train/")
    print(f"  - Confusion matrix: runs/detect/train/confusion_matrix.png")
    print(f"  - Training curves: runs/detect/train/results.png")
    print(f"  - Validation predictions: runs/detect/train/val_batch*_pred.jpg")
    print("\nNext step: Run inference_yolo.py to generate predictions!")