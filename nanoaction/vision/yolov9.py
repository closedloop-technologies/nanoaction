"""
YOLOv9+ implementation using PyTorch and HuggingFace Hub.

This module provides a simple interface to run YOLOv9+ object detection
on images using pre-trained weights from HuggingFace.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download


class YOLOv9Plus:
    """YOLOv9+ object detection model."""

    def __init__(
        self,
        model_id: str = "merve/yolov9",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize YOLOv9+ model.

        Args:
            model_id: HuggingFace model ID
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_id = model_id
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # COCO class names
        self.class_names = self._get_coco_names()

    def _load_model(self):
        """Load YOLOv9 model from HuggingFace or use ultralytics."""
        try:
            # Try to use ultralytics YOLO implementation
            from ultralytics import YOLO

            # Download weights from HuggingFace
            try:
                model_path = hf_hub_download(
                    repo_id=self.model_id,
                    filename="yolov9c.pt"
                )
            except:
                # Fallback to direct ultralytics download
                print(f"Could not download from {self.model_id}, using ultralytics default")
                model_path = "yolov9c.pt"

            model = YOLO(model_path)
            return model

        except ImportError:
            raise ImportError(
                "ultralytics package is required. Install with: pip install ultralytics"
            )

    def _get_coco_names(self) -> List[str]:
        """Get COCO class names."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[dict]:
        """
        Run inference on an image.

        Args:
            image: Input image (path, PIL Image, or numpy array)
            conf: Confidence threshold (overrides default)
            iou: IoU threshold (overrides default)

        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: float
                - class_id: int
                - class_name: str
        """
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold

        # Run inference
        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf_val = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                detections.append({
                    'bbox': box.tolist(),
                    'confidence': conf_val,
                    'class_id': cls_id,
                    'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f'class_{cls_id}'
                })

        return detections

    def draw_detections(
        self,
        image: Union[str, Path, Image.Image],
        detections: List[dict],
        output_path: Optional[Union[str, Path]] = None,
        line_width: int = 2,
        font_size: int = 12,
    ) -> Image.Image:
        """
        Draw bounding boxes on image.

        Args:
            image: Input image
            detections: List of detections from predict()
            output_path: Optional path to save annotated image
            line_width: Width of bounding box lines
            font_size: Font size for labels

        Returns:
            Annotated PIL Image
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            image = image.copy()

        draw = ImageDraw.Draw(image)

        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Draw each detection
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'
        ]

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']

            # Choose color based on class
            color = colors[det['class_id'] % len(colors)]

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Draw label
            label = f"{class_name} {confidence:.2f}"

            # Get text bounding box for background
            bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1), label, fill='white', font=font)

        # Save if output path provided
        if output_path:
            image.save(output_path)

        return image

    @torch.no_grad()
    def batch_predict(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[List[dict]]:
        """
        Run inference on multiple images.

        Args:
            images: List of input images
            conf: Confidence threshold
            iou: IoU threshold

        Returns:
            List of detection lists, one per image
        """
        return [self.predict(img, conf, iou) for img in images]

    def get_backbone_model(self):
        """
        Get the underlying PyTorch backbone model for feature extraction.
        
        Returns:
            The PyTorch model from ultralytics YOLO
        """
        # Access the underlying PyTorch model
        return self.model.model

    @torch.no_grad()
    def extract_features(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract intermediate features/embeddings from the YOLO backbone.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            layer_idx: Which layer to extract features from. If None, extracts from
                      the last backbone layer before the detection head.
        
        Returns:
            Feature tensor of shape (1, C, H, W) where C is the number of channels
            and H, W are the spatial dimensions.
        
        Example:
            >>> yolo = YOLOv9Plus()
            >>> features = yolo.extract_features("image.jpg")
            >>> print(features.shape)  # e.g., torch.Size([1, 512, 20, 20])
        """
        # Simplified approach: Use YOLO's predictor to preprocess, then extract features
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            img = Image.open(image)
            img = np.array(img)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image
        
        # Preprocess using ultralytics transforms
        from ultralytics.data.augment import LetterBox
        letterbox = LetterBox(new_shape=(640, 640), auto=True, stride=32)
        img = letterbox(image=img)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Get the underlying PyTorch model
        model = self.get_backbone_model()
        
        # Store features using a hook
        features_dict = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                # Handle both tensor and list/tuple outputs
                if isinstance(output, torch.Tensor):
                    features_dict[name] = output
                elif isinstance(output, (list, tuple)) and len(output) > 0:
                    # Store first tensor output
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            features_dict[name] = item
                            break
            return hook
        
        # Register hooks on all layers
        hooks = []
        for i, layer in enumerate(model.model):
            hook = layer.register_forward_hook(hook_fn(f'layer_{i}'))
            hooks.append(hook)
        
        try:
            # Forward pass
            _ = model(img)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Get the requested feature or the last one
            if layer_idx is not None:
                key = f'layer_{layer_idx}'
                if key in features_dict:
                    return features_dict[key]
            
            # Return the last feature map
            if features_dict:
                last_key = max(features_dict.keys(), key=lambda x: int(x.split('_')[1]))
                return features_dict[last_key]
            
            # Fallback: just return the model output
            return model(img)
            
        except Exception as e:
            # Clean up hooks on error
            for hook in hooks:
                hook.remove()
            raise e

    @torch.no_grad()
    def get_image_embeddings(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        pool: str = 'avg',
    ) -> torch.Tensor:
        """
        Extract a single embedding vector for an image using global pooling.
        This is useful for connecting to language models or other downstream tasks.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            pool: Pooling method - 'avg' for average pooling, 'max' for max pooling,
                  or 'flatten' to flatten the spatial dimensions
        
        Returns:
            Embedding tensor of shape (1, embedding_dim) where embedding_dim depends
            on the pooling method:
            - 'avg' or 'max': (1, C) where C is number of channels
            - 'flatten': (1, C*H*W)
        
        Example:
            >>> yolo = YOLOv9Plus()
            >>> embedding = yolo.get_image_embeddings("image.jpg", pool='avg')
            >>> print(embedding.shape)  # e.g., torch.Size([1, 512])
        """
        # Extract spatial features
        features = self.extract_features(image)
        
        # Handle different feature shapes
        if features.dim() == 4:
            # Standard case: (B, C, H, W)
            if pool == 'avg':
                embedding = features.mean(dim=[2, 3])
            elif pool == 'max':
                embedding = features.amax(dim=[2, 3])
            elif pool == 'flatten':
                B = features.size(0)
                embedding = features.view(B, -1)
            else:
                raise ValueError(f"Unknown pooling method: {pool}. Use 'avg', 'max', or 'flatten'")
        elif features.dim() == 3:
            # Case: (B, C, H) - pool over last dimension
            if pool == 'avg':
                embedding = features.mean(dim=2)
            elif pool == 'max':
                embedding = features.amax(dim=2)
            elif pool == 'flatten':
                B = features.size(0)
                embedding = features.view(B, -1)
            else:
                raise ValueError(f"Unknown pooling method: {pool}")
        elif features.dim() == 2:
            # Already a 2D tensor (B, C) - no pooling needed
            embedding = features
        else:
            # Fallback: flatten everything except batch dim
            B = features.size(0)
            embedding = features.view(B, -1)
        
        return embedding


def load_yolov9(
    model_id: str = "merve/yolov9",
    device: Optional[str] = None,
    **kwargs
) -> YOLOv9Plus:
    """
    Convenience function to load YOLOv9+ model.

    Args:
        model_id: HuggingFace model ID
        device: Device to run on
        **kwargs: Additional arguments for YOLOv9Plus

    Returns:
        YOLOv9Plus model instance
    """
    return YOLOv9Plus(model_id=model_id, device=device, **kwargs)
