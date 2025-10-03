"""
Evaluation Metrics for Drone Detection

This module implements comprehensive evaluation metrics including mAP, precision, 
recall, and specialized metrics for comparing baseline YOLOv7 vs Betti-filtered results.

Author: Autonomous Drone Detection Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
from typing import List, Dict, Tuple, Optional, Union
import logging
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

class DetectionEvaluator:
    """Comprehensive evaluation for object detection with Betti filtering."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names for evaluation
        """
        self.class_names = class_names or [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]
        self.num_classes = len(self.class_names)
        
        # Evaluation results storage
        self.results = {
            'baseline': {'predictions': [], 'ground_truth': []},
            'filtered': {'predictions': [], 'ground_truth': []}
        }
        
        # Performance metrics
        self.metrics = {}
        
    def add_predictions(self, predictions: List[Dict], ground_truth: List[Dict], 
                       method: str = 'baseline'):
        """
        Add predictions and ground truth for evaluation.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth dictionaries  
            method: Method name ('baseline' or 'filtered')
        """
        if method not in self.results:
            self.results[method] = {'predictions': [], 'ground_truth': []}
        
        self.results[method]['predictions'].extend(predictions)
        self.results[method]['ground_truth'].extend(ground_truth)
    
    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
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
    
    def match_predictions_to_gt(self, predictions: List[Dict], ground_truth: List[Dict],
                               iou_threshold: float = 0.5) -> Tuple[List[bool], List[int]]:
        """
        Match predictions to ground truth based on IoU threshold.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth annotations
            iou_threshold: IoU threshold for matching
            
        Returns:
            (matches, gt_matches) - matches[i] indicates if prediction i is matched,
                                  gt_matches[i] gives the GT index for prediction i (-1 if unmatched)
        """
        matches = [False] * len(predictions)
        gt_matches = [-1] * len(predictions)
        gt_used = [False] * len(ground_truth)
        
        # Sort predictions by confidence (descending)
        pred_indices = sorted(range(len(predictions)), 
                            key=lambda i: predictions[i]['confidence'], reverse=True)
        
        for pred_idx in pred_indices:
            pred = predictions[pred_idx]
            pred_class = pred['class_id']
            pred_bbox = pred['bbox']
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truth):
                if gt_used[gt_idx] or gt['class_id'] != pred_class:
                    continue
                
                iou = self.compute_iou(pred_bbox, gt['bbox'])
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Assign match if found
            if best_gt_idx >= 0:
                matches[pred_idx] = True
                gt_matches[pred_idx] = best_gt_idx
                gt_used[best_gt_idx] = True
        
        return matches, gt_matches
    
    def compute_precision_recall_curve(self, predictions: List[Dict], ground_truth: List[Dict],
                                     class_id: int, iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute precision-recall curve for a specific class.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth annotations
            class_id: Class ID to evaluate
            iou_threshold: IoU threshold for matching
            
        Returns:
            (precision, recall, ap) - precision values, recall values, average precision
        """
        # Filter predictions and ground truth for this class
        class_preds = [p for p in predictions if p['class_id'] == class_id]
        class_gt = [g for g in ground_truth if g['class_id'] == class_id]
        
        if not class_gt:
            return np.array([]), np.array([]), 0.0
        
        if not class_preds:
            return np.array([1, 0]), np.array([0, 0]), 0.0
        
        # Sort predictions by confidence
        class_preds = sorted(class_preds, key=lambda x: x['confidence'], reverse=True)
        
        # Match predictions to ground truth
        matches, _ = self.match_predictions_to_gt(class_preds, class_gt, iou_threshold)
        
        # Compute precision and recall at each threshold
        tp = np.cumsum(matches)
        fp = np.cumsum([not m for m in matches])
        
        precision = tp / (tp + fp)
        recall = tp / len(class_gt)
        
        # Add endpoint for recall=0
        precision = np.concatenate(([1], precision))
        recall = np.concatenate(([0], recall))
        
        # Compute Average Precision using interpolated precision
        ap = 0
        for i in range(len(recall) - 1):
            max_precision = max(precision[i:])
            ap += max_precision * (recall[i + 1] - recall[i])
        
        return precision, recall, ap
    
    def compute_map(self, predictions: List[Dict], ground_truth: List[Dict],
                   iou_thresholds: List[float] = None) -> Dict:
        """
        Compute mean Average Precision (mAP) across classes and IoU thresholds.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth annotations
            iou_thresholds: List of IoU thresholds (default: [0.5])
            
        Returns:
            Dictionary containing mAP metrics
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5]
        
        results = {
            'map_per_class': {},
            'map_per_iou': {},
            'overall_map': 0.0,
            'map_50': 0.0,
            'map_50_95': 0.0
        }
        
        # Compute AP for each class and IoU threshold
        all_aps = []
        
        for iou_thresh in iou_thresholds:
            class_aps = []
            
            for class_id in range(self.num_classes):
                _, _, ap = self.compute_precision_recall_curve(
                    predictions, ground_truth, class_id, iou_thresh
                )
                class_aps.append(ap)
                
                if iou_thresh not in results['map_per_class']:
                    results['map_per_class'][iou_thresh] = {}
                results['map_per_class'][iou_thresh][self.class_names[class_id]] = ap
            
            map_at_iou = np.mean(class_aps)
            results['map_per_iou'][iou_thresh] = map_at_iou
            all_aps.extend(class_aps)
        
        # Overall mAP
        results['overall_map'] = np.mean(all_aps)
        
        # Standard metrics
        if 0.5 in iou_thresholds:
            results['map_50'] = results['map_per_iou'][0.5]
        
        # mAP@0.5:0.95 (COCO style)
        coco_ious = [i/100.0 for i in range(50, 100, 5)]
        if all(iou in iou_thresholds for iou in coco_ious):
            coco_maps = [results['map_per_iou'][iou] for iou in coco_ious]
            results['map_50_95'] = np.mean(coco_maps)
        
        return results
    
    def compute_detection_metrics(self, predictions: List[Dict], ground_truth: List[Dict],
                                iou_threshold: float = 0.5) -> Dict:
        """
        Compute comprehensive detection metrics.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth annotations
            iou_threshold: IoU threshold for matching
            
        Returns:
            Dictionary containing various metrics
        """
        # Match predictions to ground truth
        matches, _ = self.match_predictions_to_gt(predictions, ground_truth, iou_threshold)
        
        # Overall metrics
        tp = sum(matches)
        fp = len(matches) - tp
        fn = len(ground_truth) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Per-class metrics
        class_metrics = {}
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            
            class_preds = [p for p in predictions if p['class_id'] == class_id]
            class_gt = [g for g in ground_truth if g['class_id'] == class_id]
            
            if class_gt:  # Only compute if GT exists for this class
                class_matches, _ = self.match_predictions_to_gt(class_preds, class_gt, iou_threshold)
                
                class_tp = sum(class_matches)
                class_fp = len(class_matches) - class_tp
                class_fn = len(class_gt) - class_tp
                
                class_precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) > 0 else 0
                class_recall = class_tp / (class_tp + class_fn) if (class_tp + class_fn) > 0 else 0
                class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
                
                class_metrics[class_name] = {
                    'precision': class_precision,
                    'recall': class_recall,
                    'f1_score': class_f1,
                    'tp': class_tp,
                    'fp': class_fp,
                    'fn': class_fn,
                    'support': len(class_gt)
                }
        
        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'support': len(ground_truth)
            },
            'per_class': class_metrics
        }
    
    def evaluate_method(self, method: str, iou_thresholds: List[float] = None) -> Dict:
        """
        Evaluate a specific method (baseline or filtered).
        
        Args:
            method: Method name ('baseline' or 'filtered')
            iou_thresholds: List of IoU thresholds
            
        Returns:
            Evaluation results dictionary
        """
        if method not in self.results:
            raise ValueError(f"Method '{method}' not found in results")
        
        predictions = self.results[method]['predictions']
        ground_truth = self.results[method]['ground_truth']
        
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.75] + [i/100.0 for i in range(50, 100, 5)]
        
        # Compute mAP
        map_results = self.compute_map(predictions, ground_truth, iou_thresholds)
        
        # Compute detection metrics at IoU=0.5
        detection_metrics = self.compute_detection_metrics(predictions, ground_truth, 0.5)
        
        # Combine results
        results = {
            'map': map_results,
            'detection_metrics': detection_metrics,
            'num_predictions': len(predictions),
            'num_ground_truth': len(ground_truth)
        }
        
        return results
    
    def compare_methods(self) -> Dict:
        """
        Compare baseline and filtered methods.
        
        Returns:
            Comparison results dictionary
        """
        comparison = {}
        
        # Evaluate each method
        for method in ['baseline', 'filtered']:
            if method in self.results and self.results[method]['predictions']:
                comparison[method] = self.evaluate_method(method)
        
        # Compute improvements
        if 'baseline' in comparison and 'filtered' in comparison:
            baseline_map = comparison['baseline']['map']['map_50']
            filtered_map = comparison['filtered']['map']['map_50']
            
            baseline_precision = comparison['baseline']['detection_metrics']['overall']['precision']
            filtered_precision = comparison['filtered']['detection_metrics']['overall']['precision']
            
            baseline_recall = comparison['baseline']['detection_metrics']['overall']['recall']
            filtered_recall = comparison['filtered']['detection_metrics']['overall']['recall']
            
            comparison['improvements'] = {
                'map_50_improvement': filtered_map - baseline_map,
                'map_50_relative_improvement': (filtered_map - baseline_map) / max(baseline_map, 1e-6),
                'precision_improvement': filtered_precision - baseline_precision,
                'recall_improvement': filtered_recall - baseline_recall,
                'detection_reduction': {
                    'absolute': comparison['baseline']['num_predictions'] - comparison['filtered']['num_predictions'],
                    'relative': (comparison['baseline']['num_predictions'] - comparison['filtered']['num_predictions']) / max(comparison['baseline']['num_predictions'], 1)
                }
            }
        
        return comparison
    
    def create_evaluation_plots(self, save_dir: str) -> Dict[str, str]:
        """
        Create evaluation plots and save them.
        
        Args:
            save_dir: Directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plot_files = {}
        
        # 1. mAP comparison bar plot
        comparison = self.compare_methods()
        
        if 'baseline' in comparison and 'filtered' in comparison:
            # mAP comparison
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            methods = ['Baseline', 'Filtered']
            map_50_values = [
                comparison['baseline']['map']['map_50'],
                comparison['filtered']['map']['map_50']
            ]
            
            bars = ax.bar(methods, map_50_values, color=['skyblue', 'lightcoral'])
            ax.set_ylabel('mAP@0.5')
            ax.set_title('mAP@0.5 Comparison: Baseline vs Betti Filtered')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, map_50_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = save_dir / 'map_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['map_comparison'] = str(plot_path)
            
            # 2. Per-class mAP comparison
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            class_names = self.class_names
            baseline_class_map = [comparison['baseline']['map']['map_per_class'][0.5][name] for name in class_names]
            filtered_class_map = [comparison['filtered']['map']['map_per_class'][0.5][name] for name in class_names]
            
            x = np.arange(len(class_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, baseline_class_map, width, label='Baseline', color='skyblue')
            bars2 = ax.bar(x + width/2, filtered_class_map, width, label='Filtered', color='lightcoral')
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('mAP@0.5')
            ax.set_title('Per-Class mAP@0.5 Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plot_path = save_dir / 'per_class_map.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['per_class_map'] = str(plot_path)
            
            # 3. Precision-Recall curves
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            for i, class_name in enumerate(class_names):
                class_id = i
                
                # Baseline PR curve
                baseline_preds = self.results['baseline']['predictions']
                baseline_gt = self.results['baseline']['ground_truth']
                baseline_precision, baseline_recall, baseline_ap = self.compute_precision_recall_curve(
                    baseline_preds, baseline_gt, class_id, 0.5
                )
                
                # Filtered PR curve
                filtered_preds = self.results['filtered']['predictions']
                filtered_gt = self.results['filtered']['ground_truth']
                filtered_precision, filtered_recall, filtered_ap = self.compute_precision_recall_curve(
                    filtered_preds, filtered_gt, class_id, 0.5
                )
                
                ax = axes[i]
                if len(baseline_precision) > 0:
                    ax.plot(baseline_recall, baseline_precision, label=f'Baseline (AP={baseline_ap:.3f})', color='blue')
                if len(filtered_precision) > 0:
                    ax.plot(filtered_recall, filtered_precision, label=f'Filtered (AP={filtered_ap:.3f})', color='red')
                
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(class_name)
                ax.legend()
                ax.grid(True)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plot_path = save_dir / 'precision_recall_curves.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['precision_recall_curves'] = str(plot_path)
            
            # 4. Detection count comparison
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            detection_counts = [
                comparison['baseline']['num_predictions'],
                comparison['filtered']['num_predictions']
            ]
            
            bars = ax.bar(methods, detection_counts, color=['skyblue', 'lightcoral'])
            ax.set_ylabel('Number of Detections')
            ax.set_title('Detection Count: Baseline vs Filtered')
            
            # Add value labels
            for bar, value in zip(bars, detection_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(detection_counts) * 0.01,
                       f'{value}', ha='center', va='bottom')
            
            # Add reduction percentage
            reduction = comparison['improvements']['detection_reduction']['relative'] * 100
            ax.text(0.5, max(detection_counts) * 0.8, f'Reduction: {reduction:.1f}%', 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            plot_path = save_dir / 'detection_count_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['detection_count_comparison'] = str(plot_path)
        
        return plot_files
    
    def save_evaluation_report(self, save_path: str):
        """
        Save a comprehensive evaluation report.
        
        Args:
            save_path: Path to save the report
        """
        comparison = self.compare_methods()
        
        report = {
            'evaluation_summary': {
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'class_names': self.class_names,
                'num_classes': self.num_classes
            },
            'results': comparison,
            'summary': {}
        }
        
        # Add summary statistics
        if 'improvements' in comparison:
            improvements = comparison['improvements']
            report['summary'] = {
                'map_50_baseline': comparison['baseline']['map']['map_50'],
                'map_50_filtered': comparison['filtered']['map']['map_50'],
                'map_50_improvement': improvements['map_50_improvement'],
                'map_50_relative_improvement_percent': improvements['map_50_relative_improvement'] * 100,
                'detection_reduction_absolute': improvements['detection_reduction']['absolute'],
                'detection_reduction_percent': improvements['detection_reduction']['relative'] * 100,
                'precision_improvement': improvements['precision_improvement'],
                'recall_improvement': improvements['recall_improvement']
            }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {save_path}")


def load_predictions_from_json(json_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load predictions and ground truth from JSON file.
    
    Args:
        json_path: Path to JSON file containing results
        
    Returns:
        (predictions, ground_truth) tuple
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    predictions = data.get('predictions', [])
    ground_truth = data.get('ground_truth', [])
    
    return predictions, ground_truth


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate drone detection results')
    parser.add_argument('--baseline', type=str, required=True,
                       help='Path to baseline results JSON')
    parser.add_argument('--filtered', type=str, required=True,
                       help='Path to filtered results JSON')
    parser.add_argument('--output', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save evaluation plots')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DetectionEvaluator()
    
    # Load baseline results
    baseline_preds, baseline_gt = load_predictions_from_json(args.baseline)
    evaluator.add_predictions(baseline_preds, baseline_gt, 'baseline')
    
    # Load filtered results
    filtered_preds, filtered_gt = load_predictions_from_json(args.filtered)
    evaluator.add_predictions(filtered_preds, filtered_gt, 'filtered')
    
    # Run evaluation
    logger.info("Computing evaluation metrics...")
    comparison = evaluator.compare_methods()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation report
    report_path = output_dir / 'evaluation_report.json'
    evaluator.save_evaluation_report(str(report_path))
    
    # Create plots if requested
    if args.save_plots:
        logger.info("Creating evaluation plots...")
        plot_files = evaluator.create_evaluation_plots(str(output_dir))
        logger.info(f"Plots saved: {list(plot_files.keys())}")
    
    # Print summary
    if 'improvements' in comparison:
        improvements = comparison['improvements']
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Baseline mAP@0.5: {comparison['baseline']['map']['map_50']:.3f}")
        print(f"Filtered mAP@0.5: {comparison['filtered']['map']['map_50']:.3f}")
        print(f"mAP Improvement: {improvements['map_50_improvement']:+.3f} ({improvements['map_50_relative_improvement']*100:+.1f}%)")
        print(f"Detection Reduction: {improvements['detection_reduction']['absolute']} ({improvements['detection_reduction']['relative']*100:.1f}%)")
        print(f"Precision Change: {improvements['precision_improvement']:+.3f}")
        print(f"Recall Change: {improvements['recall_improvement']:+.3f}")
        print("="*60)
    
    print(f"\nDetailed results saved to: {output_dir}")


if __name__ == '__main__':
    main()