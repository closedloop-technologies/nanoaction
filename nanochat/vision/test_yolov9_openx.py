"""
Test script to run YOLOv9+ on OpenX dataset images.

This script loads images from a pickle file and runs YOLOv9+ object detection,
saving the annotated images with bounding boxes.
"""

import pickle
import io
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any
import argparse

from yolov9 import load_yolov9


def load_images_from_pickle(pickle_path: str) -> List[Image.Image]:
    """
    Load images from OpenX pickle file.
    
    Args:
        pickle_path: Path to pickle file
    
    Returns:
        List of PIL Images
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    images = []
    steps = data.get('steps', [])
    
    print(f"Found {len(steps)} steps in pickle file")
    
    for i, step in enumerate(steps):
        observation = step.get('observation', {})
        img_bytes = observation.get('image')
        
        if img_bytes:
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)
    
    return images


def run_detection_on_openx(
    pickle_path: str,
    output_dir: str = "output_detections",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_images: int = None,
    device: str = None,
):
    """
    Run YOLOv9+ detection on images from OpenX pickle file.
    
    Args:
        pickle_path: Path to pickle file
        output_dir: Directory to save annotated images
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        max_images: Maximum number of images to process (None for all)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading images from {pickle_path}...")
    images = load_images_from_pickle(pickle_path)
    
    if max_images:
        images = images[:max_images]
    
    print(f"Processing {len(images)} images")
    
    # Load YOLOv9+ model
    print(f"Loading YOLOv9+ model on device: {device or 'auto'}...")
    model = load_yolov9(
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    
    # Process each image
    print("Running detections...")
    for i, image in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}...", end='\r')
        
        # Run detection
        detections = model.predict(image)
        
        # Draw and save
        output_file = output_path / f"frame_{i:04d}_detections.jpg"
        annotated_img = model.draw_detections(
            image,
            detections,
            output_path=output_file
        )
        
        # Print detection summary for first few images
        if i < 5:
            print(f"\nImage {i}: Found {len(detections)} objects")
            for det in detections[:5]:  # Show first 5 detections
                print(f"  - {det['class_name']}: {det['confidence']:.2f}")
    
    print(f"\n\nDone! Saved {len(images)} annotated images to {output_dir}/")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total images processed: {len(images)}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv9+ detection on OpenX dataset images"
    )
    parser.add_argument(
        "--pickle",
        type=str,
        default="data/openx/cmu_playing_with_food/sample_000000000074.data.pickle",
        help="Path to pickle file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_detections",
        help="Output directory for annotated images"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu, default: auto)"
    )
    
    args = parser.parse_args()
    
    run_detection_on_openx(
        pickle_path=args.pickle,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_images=args.max_images,
        device=args.device,
    )


if __name__ == "__main__":
    main()
