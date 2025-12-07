"""
YOLOv8 Inference Script for Test Images
This script runs predictions on test images and saves results in YOLO format.
"""

from ultralytics import YOLO
import os
from pathlib import Path
import cv2

# Configuration
MODEL_PATH = 'runs/detect/train/weights/best.pt'  # Path to your trained model
TEST_IMAGES_DIR = 'military_object_dataset/test/images'  # Test images folder
OUTPUT_DIR = 'predictions_txt'  # Where to save prediction TXT files
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence to save a detection (0.0-1.0)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load trained model
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Get list of test images
test_images = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
    test_images.extend(Path(TEST_IMAGES_DIR).glob(ext))

print(f"\nFound {len(test_images)} test images")
print(f"Saving predictions to: {OUTPUT_DIR}/")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"\nProcessing images...\n")

# Process each image
processed_count = 0
for img_path in test_images:
    # Run inference
    results = model(
        source=str(img_path),
        conf=CONFIDENCE_THRESHOLD,  # Confidence threshold
        iou=0.45,                   # NMS IoU threshold
        device=0,                   # GPU device
        verbose=False,              # Don't print each image
        save=False,                 # Don't save visualizations
        save_txt=False              # We'll save manually in correct format
    )
    
    # Get image dimensions for normalization
    img = cv2.imread(str(img_path))
    img_height, img_width = img.shape[:2]
    
    # Prepare output file path
    txt_filename = img_path.stem + '.txt'
    txt_path = os.path.join(OUTPUT_DIR, txt_filename)
    
    # Extract predictions and save in YOLO format
    with open(txt_path, 'w') as f:
        for result in results:
            boxes = result.boxes
            
            # Process each detection
            for i in range(len(boxes)):
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                
                # Convert to YOLO format (normalized x_center, y_center, width, height)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Get class and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Write to file in format: class_id x_center y_center width height confidence
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")
    
    processed_count += 1
    
    # Print progress every 100 images
    if processed_count % 100 == 0:
        print(f"Processed {processed_count}/{len(test_images)} images...")

print(f"\n{'='*50}")
print("INFERENCE COMPLETED!")
print(f"{'='*50}")
print(f"Total images processed: {processed_count}")
print(f"Predictions saved to: {OUTPUT_DIR}/")
print(f"\nEach .txt file contains detections in format:")
print("class_id x_center y_center width height confidence")
print(f"\nNext step: Run zip_predictions.py to create submission file!")