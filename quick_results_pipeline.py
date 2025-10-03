"""
Quick Results Pipeline for Autonomous Drone Detection

This script provides faster results by training for fewer epochs
and then running the complete evaluation and inference pipeline.

Usage:
    python quick_results_pipeline.py --epochs 10
    
Author: Autonomous Drone Detection Team
"""

import sys
import argparse
import json
import time
from pathlib import Path
import logging

# Add project root to path  
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickResultsPipeline:
    """Quick results pipeline with reduced training time."""
    
    def __init__(self):
        """Initialize the quick results pipeline."""
        self.data_dir = Path("data")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Dataset paths
        self.yolo_dir = self.data_dir / "yolo_format"
        
    def quick_train_model(self, epochs: int = 10):
        """Train YOLOv7 model quickly with fewer epochs."""
        logger.info(f"Starting quick YOLOv7 training for {epochs} epochs...")
        
        try:
            from ultralytics import YOLO
            
            # Load pretrained model
            model = YOLO('yolo11n.pt')
            
            # Quick training parameters
            train_args = {
                'data': str(self.yolo_dir / 'dataset.yaml'),
                'epochs': epochs,
                'batch': 8,
                'imgsz': 640,
                'device': 'cpu',
                'project': str(self.results_dir / 'quick_training'),
                'name': 'yolov7_quick',
                'exist_ok': True,
                'save': True,
                'val': True,
                'patience': 5,  # Early stopping
                'workers': 4,
            }
            
            logger.info(f"Quick training arguments: {train_args}")
            
            # Start training
            results = model.train(**train_args)
            
            # Get model path
            training_dir = self.results_dir / 'quick_training' / 'yolov7_quick'
            model_path = training_dir / 'weights' / 'best.pt'
            
            logger.info(f"Quick training completed! Model saved to: {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Quick training failed: {e}")
            raise
    
    def run_evaluation_and_inference(self, model_path: str):
        """Run evaluation and inference with the trained model."""
        logger.info("Running evaluation and inference...")
        
        # Import pipeline components
        from src.models.betti_filter import create_betti_filter_config
        from src.inference.detect import DroneDetectionPipeline
        
        # Use test images for evaluation
        test_images_dir = self.yolo_dir / 'test' / 'images'
        test_images = list(test_images_dir.glob('*.jpg'))[:50]  # Sample 50 images
        
        logger.info(f"Evaluating on {len(test_images)} test images")
        
        # Create pipelines
        baseline_pipeline = DroneDetectionPipeline(
            model_path=model_path,
            use_betti_filter=False
        )
        
        filtered_pipeline = DroneDetectionPipeline(
            model_path=model_path,
            use_betti_filter=True,
            filter_config=create_betti_filter_config('standard')
        )
        
        # Run evaluation
        baseline_results = []
        filtered_results = []
        
        for i, image_path in enumerate(test_images):
            if i % 10 == 0:
                logger.info(f"Processing image {i+1}/{len(test_images)}")
            
            try:
                # Baseline detection
                baseline_result = baseline_pipeline.detect_single_image(str(image_path))
                baseline_results.append(baseline_result)
                
                # Filtered detection
                filtered_result = filtered_pipeline.detect_single_image(str(image_path))
                filtered_results.append(filtered_result)
                
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                continue
        
        # Calculate results
        baseline_detections = sum(len(r['final_detections']) for r in baseline_results)
        filtered_detections = sum(len(r['final_detections']) for r in filtered_results)
        
        baseline_time = sum(r['timing']['total_time'] for r in baseline_results)
        filtered_time = sum(r['timing']['total_time'] for r in filtered_results)
        filtering_time = sum(r['timing']['filtering_time'] for r in filtered_results)
        
        # Create evaluation summary
        evaluation_summary = {
            'quick_training': {
                'model_path': model_path,
                'evaluation_images': len(baseline_results)
            },
            'baseline_results': {
                'total_detections': baseline_detections,
                'avg_detections_per_image': baseline_detections / len(baseline_results) if baseline_results else 0,
                'total_time': baseline_time,
                'fps': len(baseline_results) / baseline_time if baseline_time > 0 else 0
            },
            'filtered_results': {
                'total_detections': filtered_detections,
                'avg_detections_per_image': filtered_detections / len(filtered_results) if filtered_results else 0,
                'total_time': filtered_time,
                'fps': len(filtered_results) / filtered_time if filtered_time > 0 else 0,
                'filtering_time': filtering_time,
                'filtering_overhead': filtering_time / len(filtered_results) if filtered_results else 0
            },
            'comparison': {
                'detection_reduction': baseline_detections - filtered_detections,
                'detection_reduction_percent': ((baseline_detections - filtered_detections) / baseline_detections * 100) if baseline_detections > 0 else 0,
                'retention_rate': (filtered_detections / baseline_detections) if baseline_detections > 0 else 0,
                'speed_impact_percent': ((filtered_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
            }
        }
        
        # Save results
        eval_dir = self.results_dir / 'quick_evaluation'
        eval_dir.mkdir(exist_ok=True)
        
        with open(eval_dir / 'quick_evaluation_summary.json', 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        # Generate demo visualizations
        demo_dir = self.results_dir / 'quick_demo'
        demo_dir.mkdir(exist_ok=True)
        
        # Process 5 sample images for visualization
        sample_images = test_images[:5]
        for i, image_path in enumerate(sample_images):
            logger.info(f"Creating demo visualization {i+1}/5: {image_path.name}")
            
            try:
                result = filtered_pipeline.detect_single_image(
                    str(image_path),
                    save_results=True,
                    output_dir=str(demo_dir)
                )
                
                # Save result
                result_file = demo_dir / f"{image_path.stem}_result.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                    
            except Exception as e:
                logger.warning(f"Failed to create demo for {image_path}: {e}")
        
        # Print summary
        print("\n" + "="*60)
        print("QUICK RESULTS EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Images processed: {evaluation_summary['quick_training']['evaluation_images']}")
        print(f"Baseline detections: {evaluation_summary['baseline_results']['total_detections']}")
        print(f"Filtered detections: {evaluation_summary['filtered_results']['total_detections']}")
        print(f"Detection reduction: {evaluation_summary['comparison']['detection_reduction']} ({evaluation_summary['comparison']['detection_reduction_percent']:.1f}%)")
        print(f"Retention rate: {evaluation_summary['comparison']['retention_rate']:.2%}")
        print(f"Baseline FPS: {evaluation_summary['baseline_results']['fps']:.1f}")
        print(f"Filtered FPS: {evaluation_summary['filtered_results']['fps']:.1f}")
        print(f"Filtering overhead: {evaluation_summary['filtered_results']['filtering_overhead']*1000:.1f}ms per image")
        print(f"Demo visualizations: {demo_dir}")
        print("="*60)
        
        return evaluation_summary
    
    def run_quick_pipeline(self, epochs: int = 10):
        """Run the complete quick pipeline."""
        logger.info(f"🚀 Starting quick results pipeline with {epochs} epochs...")
        
        # Step 1: Quick training
        logger.info(f"\n🎯 Step 1: Quick training ({epochs} epochs)...")
        model_path = self.quick_train_model(epochs)
        
        # Step 2: Evaluation and inference
        logger.info("\n📈 Step 2: Running evaluation and inference...")
        evaluation_summary = self.run_evaluation_and_inference(model_path)
        
        logger.info("\n🎉 Quick pipeline completed!")
        
        return {
            'model_path': model_path,
            'evaluation_summary': evaluation_summary
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Quick Results Pipeline for Autonomous Drone Detection')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = QuickResultsPipeline()
    
    try:
        results = pipeline.run_quick_pipeline(args.epochs)
        print(f"\n✓ Quick pipeline completed!")
        print(f"  Model: {results['model_path']}")
        print(f"  Results: results/quick_evaluation/")
        print(f"  Demo: results/quick_demo/")
        
    except Exception as e:
        logger.error(f"Quick pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()