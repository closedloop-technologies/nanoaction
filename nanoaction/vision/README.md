# Vision Module

This module provides computer vision models for the nanoaction project.

## YOLOv9+

YOLOv9+ object detection implementation using PyTorch and Ultralytics.

### Features

- Downloads pre-trained weights automatically from HuggingFace or Ultralytics
- Supports both CUDA and CPU inference
- Easy-to-use API for single image or batch prediction
- Automatic bounding box visualization with class labels
- COCO dataset class names (80 classes)

### Usage

#### Basic Usage

```python
from nanochat.vision import load_yolov9

# Load model
model = load_yolov9(device='cpu')  # or 'cuda'

# Run detection on an image
detections = model.predict('path/to/image.jpg')

# Draw bounding boxes
annotated_img = model.draw_detections('path/to/image.jpg', detections, output_path='output.jpg')
```

#### Test Script

Run YOLOv9+ on OpenX dataset images:

```bash
cd nanochat/vision
uv run python test_yolov9_openx.py \
    --pickle ../../data/openx/cmu_playing_with_food/sample_000000000074.data.pickle \
    --output ../../output_detections \
    --device cpu \
    --max-images 10 \
    --conf 0.25 \
    --iou 0.45
```

#### Arguments

- `--pickle`: Path to OpenX pickle file (default: `data/openx/cmu_playing_with_food/sample_000000000074.data.pickle`)
- `--output`: Output directory for annotated images (default: `output_detections`)
- `--device`: Device to run on - `cpu` or `cuda` (default: auto-detect)
- `--max-images`: Maximum number of images to process (default: all)
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)

### API Reference

#### `YOLOv9Plus`

Main class for YOLOv9+ object detection.

**Constructor:**
```python
YOLOv9Plus(
    model_id: str = "merve/yolov9",
    device: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
)
```

**Methods:**

- `predict(image, conf=None, iou=None)` - Run inference on a single image
  - Returns: List of detections with `bbox`, `confidence`, `class_id`, `class_name`

- `draw_detections(image, detections, output_path=None)` - Draw bounding boxes on image
  - Returns: Annotated PIL Image

- `batch_predict(images, conf=None, iou=None)` - Run inference on multiple images
  - Returns: List of detection lists

#### `load_yolov9()`

Convenience function to load YOLOv9+ model.

```python
load_yolov9(
    model_id: str = "merve/yolov9",
    device: Optional[str] = None,
    **kwargs
) -> YOLOv9Plus
```

### Dependencies

- `torch>=2.8.0`
- `ultralytics>=8.0.0`
- `pillow>=12.0.0`
- `huggingface-hub>=0.34.4`
- `numpy==1.26.4`

### Example Output

The test script processes all images from the OpenX pickle file and saves them with bounding boxes drawn around detected objects. Each detection includes:

- Colored bounding box
- Class label
- Confidence score

Output files are saved as `frame_XXXX_detections.jpg` in the specified output directory.

### Performance Notes

- **CPU**: Slower but works on any machine (~2-3 seconds per image)
- **CUDA**: Much faster but requires GPU with sufficient memory (~0.1-0.2 seconds per image)
- If you encounter CUDA out of memory errors, use `--device cpu` flag

### Supported Classes

YOLOv9+ detects 80 COCO classes including:
- People and animals (person, dog, cat, bird, etc.)
- Vehicles (car, bicycle, motorcycle, etc.)
- Furniture (chair, couch, bed, dining table, etc.)
- Kitchen items (cup, fork, knife, spoon, bowl, etc.)
- Food items (banana, apple, sandwich, pizza, etc.)
- Electronics (laptop, tv, cell phone, keyboard, etc.)

See `yolov9.py` for the complete list.
