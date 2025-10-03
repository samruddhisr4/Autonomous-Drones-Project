"""
YOLOv7 Model Wrapper for VisDrone Detection

This module provides a wrapper around YOLOv7 for training and inference
on the VisDrone dataset with drone-specific optimizations.

Author: Autonomous Drone Detection Team
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import yaml
from typing import List, Tuple, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOv7DroneModel:
    """YOLOv7 model wrapper optimized for drone detection."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None,
                 img_size: int = 640,
                 conf_thresh: float = 0.25,
                 iou_thresh: float = 0.45):
        """
        Initialize YOLOv7 model.
        
        Args:
            model_path: Path to model weights (.pt file)
            config_path: Path to model configuration (.yaml file)
            device: Device to run on ('cpu', 'cuda', 'auto')
            img_size: Input image size
            conf_thresh: Confidence threshold for detections
            iou_thresh: IoU threshold for NMS
        """
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Set device
        if device == 'auto' or device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # VisDrone class names (excluding 'ignored' class)
        self.class_names = [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]
        
        self.num_classes = len(self.class_names)
        self.model = None
        self.model_path = model_path
        self.config_path = config_path
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load YOLOv7 model from checkpoint."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Load model using torch.hub or ultralytics
            if self._is_ultralytics_format(model_path):
                from ultralytics import YOLO
                self.model = YOLO(str(model_path))
                logger.info(f"Loaded Ultralytics YOLOv7 model from {model_path}")
            else:
                # Load PyTorch model directly
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                self.model = checkpoint['model'] if 'model' in checkpoint else checkpoint
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded PyTorch YOLOv7 model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _is_ultralytics_format(self, model_path: Path) -> bool:
        """Check if model is in Ultralytics format."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            return 'model' in checkpoint and hasattr(checkpoint.get('model'), 'yaml')
        except:
            return False
    
    def load_pretrained(self, model_name: str = 'yolov7'):
        """Load pretrained YOLOv7 model."""
        try:
            if model_name.startswith('yolov7'):
                # Try to load from ultralytics
                from ultralytics import YOLO
                self.model = YOLO(f'{model_name}.pt')
                logger.info(f"Loaded pretrained {model_name} model")
            else:
                # Load from torch hub
                self.model = torch.hub.load('WongKinYiu/yolov7', model_name, pretrained=True)
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded pretrained {model_name} model from torch hub")
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for YOLOv7 inference.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        h, w = img.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Pad image to square
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        
        img_padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized
        
        # Normalize and convert to tensor
        img_tensor = torch.from_numpy(img_padded).float()
        img_tensor /= 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW
        
        return img_tensor.to(self.device)
    
    def postprocess_detections(self, predictions, original_shape: Tuple[int, int]) -> List[Dict]:
        """
        Post-process YOLOv7 predictions.
        
        Args:
            predictions: Raw model predictions
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if hasattr(self.model, 'predict'):  # Ultralytics format
            # Handle Ultralytics YOLO output
            for result in predictions:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        conf = float(boxes.conf[i])
                        if conf >= self.conf_thresh:
                            # Get box coordinates (xyxy format)
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            class_id = int(boxes.cls[i])
                            
                            # Scale coordinates back to original image
                            h, w = original_shape
                            scale = min(self.img_size / h, self.img_size / w)
                            pad_h = (self.img_size - int(h * scale)) // 2
                            pad_w = (self.img_size - int(w * scale)) // 2
                            
                            x1 = (x1 - pad_w) / scale
                            y1 = (y1 - pad_h) / scale
                            x2 = (x2 - pad_w) / scale
                            y2 = (y2 - pad_h) / scale
                            
                            # Clip to image bounds
                            x1 = max(0, min(w, x1))
                            y1 = max(0, min(h, y1))
                            x2 = max(0, min(w, x2))
                            y2 = max(0, min(h, y2))
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class_id': class_id,
                                'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
                            }
                            detections.append(detection)
        else:
            # Handle PyTorch hub format (if using direct torch model)
            # This would need to be implemented based on the specific model output format
            logger.warning("PyTorch hub format postprocessing not implemented yet")
        
        return detections
    
    def predict(self, image: Union[str, np.ndarray]) -> List[Dict]:
        """
        Run inference on a single image.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            List of detections
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() or load_pretrained() first.")
        
        # Get original image shape
        if isinstance(image, str):
            original_img = cv2.imread(image)
            if original_img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            original_img = image
        
        original_shape = original_img.shape[:2]
        
        # Run inference
        with torch.no_grad():
            if hasattr(self.model, 'predict'):  # Ultralytics format
                results = self.model.predict(
                    image, 
                    imgsz=self.img_size,
                    conf=self.conf_thresh,
                    iou=self.iou_thresh,
                    verbose=False
                )
                detections = self.postprocess_detections(results, original_shape)
            else:
                # Preprocess image
                img_tensor = self.preprocess_image(image)
                
                # Run inference
                predictions = self.model(img_tensor)
                detections = self.postprocess_detections(predictions, original_shape)
        
        return detections
    
    def predict_batch(self, images: List[Union[str, np.ndarray]]) -> List[List[Dict]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of image paths or numpy arrays
            
        Returns:
            List of detection lists (one per image)
        """
        batch_detections = []
        
        for image in images:
            detections = self.predict(image)
            batch_detections.append(detections)
        
        return batch_detections
    
    def save_model(self, save_path: str):
        """Save model to file."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, 'save'):  # Ultralytics format
            self.model.save(str(save_path))
        else:
            torch.save(self.model.state_dict(), save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "device": str(self.device),
            "img_size": self.img_size,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "conf_thresh": self.conf_thresh,
            "iou_thresh": self.iou_thresh,
        }
        
        if hasattr(self.model, 'info'):
            # Ultralytics model info
            try:
                model_info = self.model.info()
                info.update(model_info)
            except:
                pass
        
        return info


# Utility functions
def create_model_config(num_classes: int = 10, 
                       img_size: int = 640,
                       save_path: Optional[str] = None) -> Dict:
    """Create YOLOv7 model configuration for VisDrone."""
    config = {
        'nc': num_classes,  # number of classes
        'depth_multiple': 1.0,  # model depth multiple
        'width_multiple': 1.0,  # layer channel multiple
        'anchors': [
            [12, 16, 19, 36, 40, 28],  # P3/8
            [36, 75, 76, 55, 72, 146],  # P4/16
            [142, 110, 192, 243, 459, 401]  # P5/32
        ],
        'backbone': [
            # [from, number, module, args]
            [-1, 1, 'Conv', [32, 3, 1]],  # 0
            [-1, 1, 'Conv', [64, 3, 2]],  # 1-P1/2
            [-1, 1, 'Conv', [64, 3, 1]],
            [-1, 1, 'Conv', [128, 3, 2]],  # 3-P2/4
            [-1, 1, 'Conv', [64, 1, 1]],
            [-2, 1, 'Conv', [64, 1, 1]],
            [-1, 1, 'Conv', [64, 3, 1]],
            [-1, 1, 'Conv', [64, 3, 1]],
            [-1, 1, 'Conv', [64, 3, 1]],
            [-1, 1, 'Conv', [64, 3, 1]],
            [[-1, -3, -5, -6], 1, 'Concat', [1]],
            [-1, 1, 'Conv', [256, 1, 1]],  # 11
            [-1, 1, 'MP', []],
            [-1, 1, 'Conv', [128, 1, 1]],
            [-3, 1, 'Conv', [128, 1, 1]],
            [-1, 1, 'Conv', [128, 3, 2]],
            [[-1, -3], 1, 'Concat', [1]],  # 16-P3/8
            [-1, 1, 'Conv', [128, 1, 1]],
            [-2, 1, 'Conv', [128, 1, 1]],
            [-1, 1, 'Conv', [128, 3, 1]],
            [-1, 1, 'Conv', [128, 3, 1]],
            [-1, 1, 'Conv', [128, 3, 1]],
            [-1, 1, 'Conv', [128, 3, 1]],
            [[-1, -3, -5, -6], 1, 'Concat', [1]],
            [-1, 1, 'Conv', [512, 1, 1]],  # 24
        ],
        'head': [
            [24, 1, 'Conv', [256, 1, 1]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 11], 1, 'Concat', [1]],  # cat backbone P4
            [-1, 1, 'Conv', [128, 1, 1]],
            [-1, 1, 'Conv', [128, 3, 1]],
            [-1, 1, 'Conv', [128, 3, 1]],
            [-1, 1, 'Conv', [128, 3, 1]],
            [-1, 1, 'Conv', [128, 3, 1]],
            [[-1, -2, -3, -4, -5], 1, 'Concat', [1]],
            [-1, 1, 'Conv', [256, 1, 1]],  # 34
            
            [-1, 1, 'Conv', [128, 1, 1]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
            [-1, 1, 'Conv', [64, 1, 1]],
            [-1, 1, 'Conv', [64, 3, 1]],
            [-1, 1, 'Conv', [64, 3, 1]],
            [-1, 1, 'Conv', [64, 3, 1]],
            [-1, 1, 'Conv', [64, 3, 1]],
            [[-1, -2, -3, -4, -5], 1, 'Concat', [1]],
            [-1, 1, 'Conv', [128, 1, 1]],  # 44
            
            [-1, 1, 'Conv', [128, 3, 2]],
            [[-1, 34], 1, 'Concat', [1]],  # cat head P4
            [-1, 1, 'Conv', [256, 1, 1]],
            [-1, 1, 'Conv', [256, 3, 1]],
            [-1, 1, 'Conv', [256, 3, 1]],
            [-1, 1, 'Conv', [256, 3, 1]],
            [-1, 1, 'Conv', [256, 3, 1]],
            [[-1, -2, -3, -4, -5], 1, 'Concat', [1]],
            [-1, 1, 'Conv', [512, 1, 1]],  # 53
            
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 24], 1, 'Concat', [1]],  # cat head P5
            [-1, 1, 'Conv', [512, 1, 1]],
            [-1, 1, 'Conv', [512, 3, 1]],
            [-1, 1, 'Conv', [512, 3, 1]],
            [-1, 1, 'Conv', [512, 3, 1]],
            [-1, 1, 'Conv', [512, 3, 1]],
            [[-1, -2, -3, -4, -5], 1, 'Concat', [1]],
            [-1, 1, 'Conv', [1024, 1, 1]],  # 62
            
            [[44, 53, 62], 1, 'Detect', [num_classes, 'anchors']],  # Detect(P3, P4, P5)
        ]
    }
    
    if save_path:
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Model config saved to {save_path}")
    
    return config