"""
Visualization Tools for Drone Detection Analysis

Author: Autonomous Drone Detection Team
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DroneDetectionVisualizer:
    """Visualization tools for drone detection analysis."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]
        self.class_colors = plt.cm.tab10(np.linspace(0, 1, len(self.class_names)))
        
    def create_detection_overlay(self, image: np.ndarray, detections: List[Dict],
                               title: str = "") -> np.ndarray:
        """Create detection overlay on image."""
        overlay = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            class_id = det['class_id']
            class_name = det.get('class_name', self.class_names[class_id])
            
            x1, y1, x2, y2 = map(int, bbox)
            color = tuple(int(c * 255) for c in self.class_colors[class_id % len(self.class_colors)][:3])
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(overlay, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(overlay, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if title:
            cv2.putText(overlay, title, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return overlay
    
    def create_before_after_comparison(self, image: np.ndarray, 
                                     original_detections: List[Dict],
                                     filtered_detections: List[Dict]) -> np.ndarray:
        """Create before/after filtering comparison."""
        h, w = image.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Left: Original detections
        left_img = self.create_detection_overlay(image.copy(), original_detections, "Before")
        comparison[:, :w] = left_img
        
        # Right: Filtered detections
        right_img = self.create_detection_overlay(image.copy(), filtered_detections, "After")
        comparison[:, w:] = right_img
        
        # Add separator and stats
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 3)
        
        stats_text = [
            f"Original: {len(original_detections)}",
            f"Filtered: {len(filtered_detections)}",
            f"Reduction: {len(original_detections) - len(filtered_detections)}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(comparison, text, (10, h - 60 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return comparison


def create_summary_visualization(results_dir: str, output_path: str):
    """Create comprehensive summary visualization."""
    results_dir = Path(results_dir)
    
    # Load evaluation data
    evaluation_file = results_dir / 'evaluation_report.json'
    if not evaluation_file.exists():
        logger.warning("No evaluation report found")
        return
    
    with open(evaluation_file, 'r') as f:
        evaluation_data = json.load(f)
    
    # Create summary plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: mAP comparison
    if 'results' in evaluation_data:
        baseline_map = evaluation_data['results']['baseline']['map']['map_50']
        filtered_map = evaluation_data['results']['filtered']['map']['map_50']
        
        methods = ['Baseline\nYOLOv7', 'YOLOv7 +\nBetti Filter']
        map_values = [baseline_map, filtered_map]
        
        ax1.bar(methods, map_values, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('mAP@0.5')
        ax1.set_title('Detection Performance')
        ax1.set_ylim(0, 1)
    
    # Plot 2: Detection reduction
    if 'summary' in evaluation_data:
        reduction_pct = evaluation_data['summary']['detection_reduction_percent']
        
        sizes = [100 - reduction_pct, reduction_pct]
        labels = ['Retained', 'Filtered']
        colors = ['lightgreen', 'lightcoral']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Detection Filtering')
    
    # Plot 3: Precision vs Recall
    if 'results' in evaluation_data:
        baseline = evaluation_data['results']['baseline']['detection_metrics']['overall']
        filtered = evaluation_data['results']['filtered']['detection_metrics']['overall']
        
        ax3.scatter(baseline['recall'], baseline['precision'], s=100, label='Baseline', color='blue')
        ax3.scatter(filtered['recall'], filtered['precision'], s=100, label='Filtered', color='red')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Recall')
        ax3.legend()
        ax3.grid(True)
    
    # Plot 4: Performance metrics
    if 'summary' in evaluation_data:
        metrics = ['mAP Improvement (%)', 'Detection Reduction (%)']
        values = [
            evaluation_data['summary']['map_50_relative_improvement_percent'],
            evaluation_data['summary']['detection_reduction_percent']
        ]
        
        colors = ['green', 'orange']
        ax4.bar(metrics, values, color=colors)
        ax4.set_ylabel('Percentage')
        ax4.set_title('Key Improvements')
    
    plt.suptitle('Autonomous Drone Detection Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Summary visualization saved to {output_path}")


def main():
    """Main visualization script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create drone detection visualizations')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/visualizations')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / 'summary_visualization.png'
    create_summary_visualization(args.results_dir, str(summary_path))
    
    print(f"✓ Visualizations created in {output_dir}")


if __name__ == '__main__':
    main()