"""
Bounding Box Utilities

Utility functions for bounding box operations, transformations, and calculations.

Author: Autonomous Drone Detection Team
"""

import numpy as np
from typing import List, Tuple, Union
import cv2

def xyxy_to_xywh(bbox: List[float]) -> List[float]:
    """Convert [x1, y1, x2, y2] to [x, y, width, height] format."""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]

def xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """Convert [x, y, width, height] to [x1, y1, x2, y2] format."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def normalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """Normalize bounding box coordinates to [0, 1] range."""
    x1, y1, x2, y2 = bbox
    return [x1/img_width, y1/img_height, x2/img_width, y2/img_height]

def denormalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """Denormalize bounding box coordinates from [0, 1] range."""
    x1, y1, x2, y2 = bbox
    return [x1*img_width, y1*img_height, x2*img_width, y2*img_height]

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def compute_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Compute center coordinates of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def compute_bbox_area(bbox: List[float]) -> float:
    """Compute area of bounding box."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def scale_bbox(bbox: List[float], scale_factor: float) -> List[float]:
    """Scale bounding box by given factor around its center."""
    x1, y1, x2, y2 = bbox
    cx, cy = compute_bbox_center(bbox)
    w, h = x2 - x1, y2 - y1
    
    new_w, new_h = w * scale_factor, h * scale_factor
    new_x1 = cx - new_w / 2
    new_y1 = cy - new_h / 2
    new_x2 = cx + new_w / 2
    new_y2 = cy + new_h / 2
    
    return [new_x1, new_y1, new_x2, new_y2]

def clip_bbox_to_image(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """Clip bounding box coordinates to image boundaries."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(img_width, x1))
    y1 = max(0, min(img_height, y1))
    x2 = max(0, min(img_width, x2))
    y2 = max(0, min(img_height, y2))
    return [x1, y1, x2, y2]

def filter_small_boxes(detections: List[dict], min_area: float = 0) -> List[dict]:
    """Filter out detections with bounding boxes smaller than min_area."""
    filtered = []
    for det in detections:
        area = compute_bbox_area(det['bbox'])
        if area >= min_area:
            filtered.append(det)
    return filtered

def non_max_suppression(detections: List[dict], iou_threshold: float = 0.5) -> List[dict]:
    """Apply Non-Maximum Suppression to remove overlapping detections."""
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while detections:
        # Take the detection with highest confidence
        current = detections.pop(0)
        keep.append(current)
        
        # Remove overlapping detections
        remaining = []
        for det in detections:
            iou = compute_iou(current['bbox'], det['bbox'])
            if iou < iou_threshold:
                remaining.append(det)
        
        detections = remaining
    
    return keep