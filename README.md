# Military Object Detection with YOLOv8

This project implements military object detection using YOLOv8 trained on a custom dataset.

## Dataset

- **Training images**: 10,000
- **Validation images**: 2,941
- **Test images**: Available in test/images

### Classes (12 total)
1. camouflage_soldier
2. weapon
3. military_tank
4. military_truck
5. military_vehicle
6. civilian
7. soldier
8. civilian_vehicle
9. military_artillery
10. trench
11. military_aircraft
12. military_warship

## Training

The model was trained using YOLOv8n (nano) for 68 epochs with the following settings:
- Batch size: 16
- Image size: 640x640
- GPU: NVIDIA GeForce GTX 1650 (4GB)
- Optimizer: SGD with cosine learning rate scheduler
- Early stopping patience: 20 epochs

### Training Results
- Best mAP50: 51.1% (epoch 68)
- Initial mAP50: 23.2% (epoch 1)

## Files

- `train_yolo.py` - Training script
- `inference_yolo.py` - Inference script
- `military_dataset.yaml` - Dataset configuration
- `dataset.md` - Dataset documentation
- `train/labels/` - Training labels (YOLO format)
- `val/labels/` - Validation labels (YOLO format)

## Requirements

```bash
pip install ultralytics torch torchvision opencv-python
```

## Usage

### Training
```bash
python train_yolo.py
```

### Inference
```bash
python inference_yolo.py
```

## Model Performance

The trained model achieves good detection accuracy on military objects. Best model weights are saved in `runs/detect/train/weights/best.pt` (not included in repository due to size).

## Notes

- Images are not included in this repository due to size constraints
- Model weights (.pt files) are excluded from version control
- Training can be resumed from checkpoints
