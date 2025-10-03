"""
Full Dataset Pipeline for Autonomous Drone Detection

This script demonstrates how to run the complete YOLOv7 + Betti filtering pipeline
on the full VisDrone dataset. Currently using validation set as example.

Usage:
    python full_dataset_pipeline.py --mode [train|evaluate|inference|convert|full]
    
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

from src.models.yolov7_model import YOLOv7DroneModel
from src.models.betti_filter import BettiNumberFilter, create_betti_filter_config
from src.evaluation.metrics import DetectionEvaluator
from src.evaluation.visualize import create_summary_visualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FullDatasetPipeline:
    """Complete pipeline for processing the full VisDrone dataset."""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """Initialize the full dataset pipeline."""
        self.config_path = config_path
        self.data_dir = Path("data")
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Dataset paths
        self.visdrone_dir = self.data_dir / "visdrone" 
        self.yolo_dir = self.data_dir / "yolo_format"
        
        # Available splits
        self.available_splits = self._check_available_splits()
        logger.info(f"Available dataset splits: {self.available_splits}")
        
    def _check_available_splits(self):
        """Check which dataset splits are available."""
        splits = []
        for split in ['train', 'val', 'test']:
            split_dir = self.visdrone_dir / split
            if split_dir.exists() and (split_dir / 'images').exists():
                image_count = len(list((split_dir / 'images').glob('*.jpg')))
                if image_count > 0:
                    splits.append(split)
                    logger.info(f"  {split}: {image_count} images")
        return splits
    
    def convert_datasets(self):
        """Convert all available VisDrone datasets to YOLO format."""
        logger.info("Converting VisDrone datasets to YOLO format...")
        
        from src.data_processing.visdrone_converter import VisDroneYOLOConverter
        
        converter = VisDroneYOLOConverter(
            visdrone_dir=str(self.visdrone_dir),
            output_dir=str(self.yolo_dir)
        )
        
        # Convert all available splits
        total_stats = {}
        for split in self.available_splits:
            try:
                logger.info(f"Converting {split} split...")
                stats = converter.convert_split(split)
                total_stats[split] = stats
                logger.info(f"✓ {split} conversion completed: {stats['processed']} images, {stats['annotations']} annotations")
            except Exception as e:
                logger.error(f"✗ Failed to convert {split}: {e}")
        
        # Create dataset YAML
        converter.create_dataset_yaml()
        
        # Save conversion stats
        stats_file = self.results_dir / "conversion_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(total_stats, f, indent=2)
        
        logger.info(f"Conversion completed. Stats saved to: {stats_file}")
        return total_stats
    
    def train_model(self, use_pretrained: bool = True):
        """Train YOLOv7 model on the dataset."""
        logger.info("Starting YOLOv7 training...")
        
        if 'train' not in self.available_splits:
            logger.warning("Training split not available. Using validation split for demo training.")
            train_split = 'val'
        else:
            train_split = 'train'
        
        val_split = 'val' if 'val' in self.available_splits else train_split
        
        try:
            from ultralytics import YOLO
            
            # Load pretrained model
            if use_pretrained:
                model = YOLO('yolo11n.pt')  # Start with pretrained weights
            else:
                model = YOLO('yolo11n.yaml')  # Train from scratch
            
            # Training parameters (adjusted for available data)
            train_args = {
                'data': str(self.yolo_dir / 'dataset.yaml'),
                'epochs': 50 if train_split == 'val' else 100,  # Fewer epochs for demo
                'batch': 8,  # Smaller batch for stability
                'imgsz': 640,
                'device': 'cpu',  # Use CPU since CUDA not available
                'project': str(self.results_dir / 'training'),
                'name': 'yolov7_visdrone_full',
                'exist_ok': True,
                'save': True,
                'val': True,
                'patience': 10,
                'workers': 4,
            }
            
            logger.info(f"Training on {train_split} split, validating on {val_split} split")
            logger.info(f"Training arguments: {train_args}")
            
            # Start training
            results = model.train(**train_args)
            
            # Save training results
            training_dir = self.results_dir / 'training' / 'yolov7_visdrone_full'
            logger.info(f"Training completed! Results saved to: {training_dir}")
            
            return str(training_dir / 'weights' / 'best.pt')
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, model_path: str):
        """Evaluate model with and without Betti filtering."""
        logger.info("Starting comprehensive model evaluation...")
        
        # Use available test split or validation split
        test_split = 'test' if 'test' in self.available_splits else 'val'
        test_images_dir = self.yolo_dir / test_split / 'images'
        
        if not test_images_dir.exists():
            logger.error(f"Test images directory not found: {test_images_dir}")
            return
        
        # Get test images
        test_images = list(test_images_dir.glob('*.jpg'))[:100]  # Limit for demo
        logger.info(f"Evaluating on {len(test_images)} images from {test_split} split")
        
        # Initialize pipeline components
        from src.inference.detect import DroneDetectionPipeline
        
        # Baseline pipeline (no Betti filtering)
        baseline_pipeline = DroneDetectionPipeline(
            model_path=model_path,
            use_betti_filter=False
        )
        
        # Filtered pipeline (with Betti filtering)
        filtered_pipeline = DroneDetectionPipeline(
            model_path=model_path,
            use_betti_filter=True,
            filter_config=create_betti_filter_config('standard')
        )
        
        # Run evaluation
        baseline_results = []
        filtered_results = []
        
        logger.info("Running baseline evaluation...")
        for i, image_path in enumerate(test_images):
            if i % 20 == 0:
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
        
        # Aggregate results
        baseline_detections = sum(len(r['final_detections']) for r in baseline_results)
        filtered_detections = sum(len(r['final_detections']) for r in filtered_results)
        
        baseline_time = sum(r['timing']['total_time'] for r in baseline_results)
        filtered_time = sum(r['timing']['total_time'] for r in filtered_results)
        filtering_time = sum(r['timing']['filtering_time'] for r in filtered_results)
        
        # Create evaluation summary
        evaluation_summary = {
            'dataset_info': {
                'test_split': test_split,
                'images_processed': len(baseline_results),
                'total_available_images': len(test_images)
            },
            'baseline_results': {
                'total_detections': baseline_detections,
                'avg_detections_per_image': baseline_detections / len(baseline_results) if baseline_results else 0,
                'total_time': baseline_time,
                'avg_time_per_image': baseline_time / len(baseline_results) if baseline_results else 0,
                'fps': len(baseline_results) / baseline_time if baseline_time > 0 else 0
            },
            'filtered_results': {
                'total_detections': filtered_detections,
                'avg_detections_per_image': filtered_detections / len(filtered_results) if filtered_results else 0,
                'total_time': filtered_time,
                'avg_time_per_image': filtered_time / len(filtered_results) if filtered_results else 0,
                'fps': len(filtered_results) / filtered_time if filtered_time > 0 else 0,
                'filtering_time': filtering_time,
                'filtering_overhead': filtering_time / len(filtered_results) if filtered_results else 0
            },
            'comparison': {
                'detection_reduction': baseline_detections - filtered_detections,
                'detection_reduction_percent': ((baseline_detections - filtered_detections) / baseline_detections * 100) if baseline_detections > 0 else 0,
                'retention_rate': (filtered_detections / baseline_detections) if baseline_detections > 0 else 0,
                'speed_impact': filtered_time - baseline_time,
                'speed_impact_percent': ((filtered_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
            }
        }
        
        # Save evaluation results
        eval_dir = self.results_dir / 'evaluation_full'
        eval_dir.mkdir(exist_ok=True)
        
        with open(eval_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        
        # Save individual results for detailed analysis
        with open(eval_dir / 'baseline_results.json', 'w') as f:
            json.dump(baseline_results, f, indent=2, default=str)
        
        with open(eval_dir / 'filtered_results.json', 'w') as f:
            json.dump(filtered_results, f, indent=2, default=str)
        
        # Print summary
        logger.info("🎉 Evaluation completed!")
        logger.info(f"Results saved to: {eval_dir}")
        
        print("\n" + "="*60)
        print("FULL DATASET EVALUATION SUMMARY")
        print("="*60)
        print(f"Images processed: {evaluation_summary['dataset_info']['images_processed']}")
        print(f"Baseline detections: {evaluation_summary['baseline_results']['total_detections']}")
        print(f"Filtered detections: {evaluation_summary['filtered_results']['total_detections']}")
        print(f"Detection reduction: {evaluation_summary['comparison']['detection_reduction']} ({evaluation_summary['comparison']['detection_reduction_percent']:.1f}%)")
        print(f"Retention rate: {evaluation_summary['comparison']['retention_rate']:.2%}")
        print(f"Baseline FPS: {evaluation_summary['baseline_results']['fps']:.1f}")
        print(f"Filtered FPS: {evaluation_summary['filtered_results']['fps']:.1f}")
        print(f"Filtering overhead: {evaluation_summary['filtered_results']['filtering_overhead']*1000:.1f}ms per image")
        print("="*60)
        
        return evaluation_summary
    
    def run_inference_demo(self, model_path: str, num_samples: int = 10):
        """Run inference demo on sample images."""
        logger.info(f"Running inference demo on {num_samples} sample images...")
        
        # Get sample images
        val_images_dir = self.yolo_dir / 'val' / 'images'
        if not val_images_dir.exists():
            logger.error("Validation images not found. Run conversion first.")
            return
        
        sample_images = list(val_images_dir.glob('*.jpg'))[:num_samples]
        
        from src.inference.detect import DroneDetectionPipeline
        
        # Create pipeline with Betti filtering
        pipeline = DroneDetectionPipeline(
            model_path=model_path,
            use_betti_filter=True,
            filter_config=create_betti_filter_config('standard')
        )
        
        # Process samples
        demo_dir = self.results_dir / 'inference_demo'
        demo_dir.mkdir(exist_ok=True)
        
        for i, image_path in enumerate(sample_images):
            logger.info(f"Processing sample {i+1}/{len(sample_images)}: {image_path.name}")
            
            try:
                result = pipeline.detect_single_image(
                    str(image_path),
                    save_results=True,
                    output_dir=str(demo_dir)
                )
                
                # Save individual result
                result_file = demo_dir / f"{image_path.stem}_result.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                    
            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
        
        logger.info(f"Inference demo completed. Results saved to: {demo_dir}")
        return demo_dir
    
    def run_full_pipeline(self):
        """Run the complete pipeline: convert -> train -> evaluate."""
        logger.info("🚀 Starting full dataset pipeline...")
        
        # Step 1: Convert datasets
        logger.info("\n📊 Step 1: Converting datasets...")
        self.convert_datasets()
        
        # Step 2: Train model (skip if model exists)
        model_path = self.results_dir / 'training' / 'yolov7_visdrone_full' / 'weights' / 'best.pt'
        
        if not model_path.exists():
            logger.info("\n🎯 Step 2: Training model...")
            model_path = self.train_model()
        else:
            logger.info(f"\n🎯 Step 2: Using existing model: {model_path}")
            model_path = str(model_path)
        
        # Step 3: Evaluate model
        logger.info("\n📈 Step 3: Evaluating model...")
        evaluation_summary = self.evaluate_model(model_path)
        
        # Step 4: Inference demo
        logger.info("\n🎨 Step 4: Running inference demo...")
        demo_dir = self.run_inference_demo(model_path)
        
        logger.info("\n🎉 Full pipeline completed successfully!")
        
        return {
            'model_path': model_path,
            'evaluation_summary': evaluation_summary,
            'demo_dir': str(demo_dir)
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Full Dataset Pipeline for Autonomous Drone Detection')
    parser.add_argument('--mode', choices=['convert', 'train', 'evaluate', 'inference', 'full'], 
                       default='full', help='Pipeline mode to run')
    parser.add_argument('--model', type=str, help='Path to trained model (for evaluate/inference modes)')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples for inference demo')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FullDatasetPipeline()
    
    try:
        if args.mode == 'convert':
            pipeline.convert_datasets()
            
        elif args.mode == 'train':
            model_path = pipeline.train_model()
            print(f"✓ Training completed! Model saved to: {model_path}")
            
        elif args.mode == 'evaluate':
            if not args.model:
                logger.error("Model path required for evaluation mode")
                return
            pipeline.evaluate_model(args.model)
            
        elif args.mode == 'inference':
            if not args.model:
                logger.error("Model path required for inference mode")
                return
            pipeline.run_inference_demo(args.model, args.samples)
            
        elif args.mode == 'full':
            results = pipeline.run_full_pipeline()
            print(f"\n✓ Full pipeline completed!")
            print(f"  Model: {results['model_path']}")
            print(f"  Demo results: {results['demo_dir']}")
            
        else:
            logger.error(f"Unknown mode: {args.mode}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()