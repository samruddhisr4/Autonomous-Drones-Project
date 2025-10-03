"""
Detection and Post-processing Pipeline

This module integrates YOLOv7 inference with Betti number filtering
for improved drone detection with reduced false positives.

Author: Autonomous Drone Detection Team
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Optional, Union
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.yolov7_model import YOLOv7DroneModel
from src.models.betti_filter import BettiNumberFilter, AdvancedBettiFilter, create_betti_filter_config

logger = logging.getLogger(__name__)

class DroneDetectionPipeline:
    """Complete detection pipeline with YOLOv7 + Betti filtering."""
    
    def __init__(self, 
                 model_path: str,
                 filter_config: Optional[Dict] = None,
                 device: str = 'auto',
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 use_betti_filter: bool = True,
                 filter_type: str = 'standard'):
        """
        Initialize detection pipeline.
        
        Args:
            model_path: Path to trained YOLOv7 model
            filter_config: Betti filter configuration
            device: Device for inference
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            use_betti_filter: Whether to apply Betti filtering
            filter_type: Type of Betti filter ('standard', 'advanced')
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_betti_filter = use_betti_filter
        
        # Initialize YOLOv7 model
        self.yolo_model = YOLOv7DroneModel(
            model_path=model_path,
            device=device,
            conf_thresh=confidence_threshold,
            iou_thresh=iou_threshold
        )
        
        # Initialize Betti filter
        if self.use_betti_filter:
            if filter_config is None:
                filter_config = create_betti_filter_config(filter_type)
            
            if filter_type == 'advanced':
                self.betti_filter = AdvancedBettiFilter(**filter_config)
            else:
                self.betti_filter = BettiNumberFilter(**filter_config)
        else:
            self.betti_filter = None
        
        # Performance tracking
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'filtered_detections': 0,
            'inference_time': 0,
            'filtering_time': 0,
            'total_time': 0
        }
        
        logger.info(f"Initialized detection pipeline with model: {model_path}")
        logger.info(f"Betti filtering: {'enabled' if use_betti_filter else 'disabled'}")
    
    def detect_single_image(self, image_path: str, save_results: bool = False,
                           output_dir: Optional[str] = None) -> Dict:
        """
        Detect objects in a single image.
        
        Args:
            image_path: Path to input image
            save_results: Whether to save detection results
            output_dir: Directory to save results
            
        Returns:
            Detection results dictionary
        """
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # YOLOv7 inference
        inference_start = time.time()
        detections = self.yolo_model.predict(image_path)
        inference_time = time.time() - inference_start
        
        original_detections = detections.copy()
        filtering_info = None
        
        # Apply Betti filtering
        if self.use_betti_filter and detections:
            filtering_start = time.time()
            filtered_detections, filtering_info = self.betti_filter.filter_detections(detections)
            filtering_time = time.time() - filtering_start
            detections = filtered_detections
        else:
            filtering_time = 0
        
        total_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_images'] += 1
        self.stats['total_detections'] += len(original_detections)
        self.stats['filtered_detections'] += len(detections)
        self.stats['inference_time'] += inference_time
        self.stats['filtering_time'] += filtering_time
        self.stats['total_time'] += total_time
        
        # Prepare results
        results = {
            'image_path': image_path,
            'original_detections': original_detections,
            'final_detections': detections,
            'filtering_info': filtering_info,
            'timing': {
                'inference_time': inference_time,
                'filtering_time': filtering_time,
                'total_time': total_time
            },
            'stats': {
                'original_count': len(original_detections),
                'final_count': len(detections),
                'filtered_count': len(original_detections) - len(detections)
            }
        }
        
        # Save results if requested
        if save_results and output_dir:
            self.save_detection_results(results, output_dir, image)
        
        return results
    
    def detect_batch(self, image_paths: List[str], output_dir: Optional[str] = None,
                    save_individual: bool = False) -> List[Dict]:
        """
        Detect objects in a batch of images.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save results
            save_individual: Whether to save individual results
            
        Returns:
            List of detection results
        """
        logger.info(f"Processing batch of {len(image_paths)} images")
        
        batch_results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            try:
                results = self.detect_single_image(
                    image_path, 
                    save_results=save_individual,
                    output_dir=output_dir
                )
                batch_results.append(results)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                continue
        
        # Save batch summary
        if output_dir:
            self.save_batch_summary(batch_results, output_dir)
        
        return batch_results
    
    def save_detection_results(self, results: Dict, output_dir: str, image: np.ndarray):
        """Save detection results including visualizations and JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(results['image_path']).stem
        
        # Save JSON results
        json_path = output_dir / f"{image_name}_results.json"
        json_results = {
            'image_path': results['image_path'],
            'timing': results['timing'],
            'stats': results['stats'],
            'detections': results['final_detections'],
            'filtering_info': results['filtering_info'] if results['filtering_info'] else {}
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Create visualizations
        self.create_visualizations(results, output_dir, image, image_name)
        
        logger.info(f"Results saved to {output_dir}")
    
    def create_visualizations(self, results: Dict, output_dir: Path, 
                            image: np.ndarray, image_name: str):
        """Create detection visualizations."""
        
        # Visualization 1: Original detections
        vis_original = self.draw_detections(
            image.copy(), 
            results['original_detections'],
            title="Original YOLOv7 Detections"
        )
        cv2.imwrite(str(output_dir / f"{image_name}_original.jpg"), vis_original)
        
        # Visualization 2: Final detections after filtering
        vis_final = self.draw_detections(
            image.copy(), 
            results['final_detections'],
            title="After Betti Filtering"
        )
        cv2.imwrite(str(output_dir / f"{image_name}_filtered.jpg"), vis_final)
        
        # Visualization 3: Betti clustering visualization
        if self.use_betti_filter and results['filtering_info']:
            vis_clusters = self.betti_filter.visualize_clusters(
                results['original_detections'],
                results['filtering_info'],
                image.copy()
            )
            cv2.imwrite(str(output_dir / f"{image_name}_clusters.jpg"), vis_clusters)
        
        # Visualization 4: Side-by-side comparison
        comparison = self.create_comparison_visualization(
            image, results['original_detections'], results['final_detections']
        )
        cv2.imwrite(str(output_dir / f"{image_name}_comparison.jpg"), comparison)
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       title: str = "") -> np.ndarray:
        """Draw detection bounding boxes on image."""
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add title
        if title:
            cv2.putText(image, title, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add detection count
        count_text = f"Detections: {len(detections)}"
        cv2.putText(image, count_text, (10, image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    def create_comparison_visualization(self, image: np.ndarray, 
                                     original_dets: List[Dict], 
                                     filtered_dets: List[Dict]) -> np.ndarray:
        """Create side-by-side comparison visualization."""
        h, w = image.shape[:2]
        
        # Create comparison image
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Left side: original detections
        left_img = self.draw_detections(image.copy(), original_dets, "Original")
        comparison[:, :w] = left_img
        
        # Right side: filtered detections
        right_img = self.draw_detections(image.copy(), filtered_dets, "Filtered")
        comparison[:, w:] = right_img
        
        # Add separator line
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
        
        # Add statistics
        stats_text = [
            f"Original: {len(original_dets)} detections",
            f"Filtered: {len(filtered_dets)} detections",
            f"Removed: {len(original_dets) - len(filtered_dets)} detections",
            f"Retention: {len(filtered_dets)/max(1,len(original_dets))*100:.1f}%"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(comparison, text, (10, h - 80 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return comparison
    
    def save_batch_summary(self, batch_results: List[Dict], output_dir: str):
        """Save batch processing summary."""
        output_dir = Path(output_dir)
        
        # Aggregate statistics
        total_original = sum(r['stats']['original_count'] for r in batch_results)
        total_final = sum(r['stats']['final_count'] for r in batch_results)
        total_filtered = total_original - total_final
        
        avg_inference_time = np.mean([r['timing']['inference_time'] for r in batch_results])
        avg_filtering_time = np.mean([r['timing']['filtering_time'] for r in batch_results])
        avg_total_time = np.mean([r['timing']['total_time'] for r in batch_results])
        
        summary = {
            'batch_size': len(batch_results),
            'total_detections': {
                'original': int(total_original),
                'final': int(total_final),
                'filtered': int(total_filtered),
                'retention_rate': total_final / max(1, total_original)
            },
            'timing': {
                'avg_inference_time': float(avg_inference_time),
                'avg_filtering_time': float(avg_filtering_time),
                'avg_total_time': float(avg_total_time),
                'total_batch_time': sum(r['timing']['total_time'] for r in batch_results)
            },
            'betti_filter_stats': self.betti_filter.get_statistics() if self.betti_filter else {}
        }
        
        # Save summary JSON
        summary_path = output_dir / 'batch_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch summary saved to {summary_path}")
        logger.info(f"Processed {len(batch_results)} images")
        logger.info(f"Detection retention rate: {summary['total_detections']['retention_rate']:.2%}")
        logger.info(f"Average processing time: {avg_total_time:.3f}s per image")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = self.stats.copy()
        
        if stats['total_images'] > 0:
            stats['avg_inference_time'] = stats['inference_time'] / stats['total_images']
            stats['avg_filtering_time'] = stats['filtering_time'] / stats['total_images']
            stats['avg_total_time'] = stats['total_time'] / stats['total_images']
            stats['fps'] = stats['total_images'] / stats['total_time'] if stats['total_time'] > 0 else 0
        
        if stats['total_detections'] > 0:
            stats['filter_rate'] = (stats['total_detections'] - stats['filtered_detections']) / stats['total_detections']
            stats['retention_rate'] = stats['filtered_detections'] / stats['total_detections']
        
        if self.betti_filter:
            stats['betti_stats'] = self.betti_filter.get_statistics()
        
        return stats


def main():
    """Main detection script."""
    parser = argparse.ArgumentParser(description='Drone detection with YOLOv7 + Betti filtering')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLOv7 model')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, default='results/predictions',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for inference (auto, cpu, cuda)')
    parser.add_argument('--no-betti', action='store_true',
                       help='Disable Betti number filtering')
    parser.add_argument('--filter-type', type=str, choices=['standard', 'advanced'], 
                       default='standard', help='Type of Betti filter')
    parser.add_argument('--save-vis', action='store_true',
                       help='Save visualization images')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DroneDetectionPipeline(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        use_betti_filter=not args.no_betti,
        filter_type=args.filter_type
    )
    
    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Single image
        logger.info(f"Processing single image: {input_path}")
        results = pipeline.detect_single_image(
            str(input_path), 
            save_results=args.save_vis,
            output_dir=str(output_dir)
        )
        
        print(f"✓ Processed {input_path.name}")
        print(f"  Original detections: {results['stats']['original_count']}")
        print(f"  Final detections: {results['stats']['final_count']}")
        print(f"  Processing time: {results['timing']['total_time']:.3f}s")
        
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in input_path.rglob('*') 
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            logger.error(f"No images found in {input_path}")
            return
        
        logger.info(f"Processing directory: {input_path} ({len(image_paths)} images)")
        batch_results = pipeline.detect_batch(
            image_paths,
            output_dir=str(output_dir),
            save_individual=args.save_vis
        )
        
        # Print summary
        total_original = sum(r['stats']['original_count'] for r in batch_results)
        total_final = sum(r['stats']['final_count'] for r in batch_results)
        avg_time = np.mean([r['timing']['total_time'] for r in batch_results])
        
        print(f"✓ Processed {len(batch_results)} images")
        print(f"  Total original detections: {total_original}")
        print(f"  Total final detections: {total_final}")
        print(f"  Average processing time: {avg_time:.3f}s per image")
        print(f"  Detection retention rate: {total_final/max(1,total_original)*100:.1f}%")
    
    else:
        logger.error(f"Input path not found: {input_path}")
        return
    
    # Print performance statistics
    stats = pipeline.get_performance_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Images processed: {stats['total_images']}")
    print(f"  Average inference time: {stats.get('avg_inference_time', 0):.3f}s")
    if pipeline.use_betti_filter:
        print(f"  Average filtering time: {stats.get('avg_filtering_time', 0):.3f}s")
    print(f"  Average FPS: {stats.get('fps', 0):.1f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()