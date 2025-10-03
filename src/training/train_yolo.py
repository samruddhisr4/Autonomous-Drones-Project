"""
YOLOv7 Training Script for VisDrone Dataset

This script handles training YOLOv7 on the VisDrone dataset with 
drone-specific optimizations and configurations.

Usage:
    python src/training/train_yolo.py --config configs/training_config.yaml
    
Author: Autonomous Drone Detection Team
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.yolov7_model import YOLOv7DroneModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv7Trainer:
    """YOLOv7 trainer for VisDrone dataset."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = self.load_config(config_path)
        self.setup_directories()
        self.setup_device()
        self.setup_logging()
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision training
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.training_stats = {
            'train_loss': [],
            'val_loss': [],
            'val_map': [],
            'learning_rates': []
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def setup_directories(self):
        """Create necessary directories for training."""
        # Create output directories
        self.output_dir = Path(self.config['output']['project']) / self.config['output']['name']
        self.output_dir.mkdir(parents=True, exist_ok=self.config['output']['exist_ok'])
        
        self.weights_dir = self.output_dir / 'weights'
        self.logs_dir = self.output_dir / 'logs'
        self.plots_dir = self.output_dir / 'plots'
        
        for dir_path in [self.weights_dir, self.logs_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_device(self):
        """Setup training device."""
        device_config = self.config['training']['device']
        
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_config)
        
        logger.info(f"Using device: {self.device}")
        
        # Setup for multi-GPU training if available
        self.world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        logger.info(f"Available GPUs: {self.world_size}")
    
    def setup_logging(self):
        """Setup training logging."""
        # Setup file logging
        log_file = self.logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Setup Weights & Biases if enabled
        if self.config['logging']['wandb']:
            wandb.init(
                project=self.config['output']['name'],
                config=self.config,
                dir=str(self.logs_dir)
            )
            logger.info("Weights & Biases logging enabled")
    
    def setup_model(self):
        """Setup YOLOv7 model for training."""
        try:
            # Try to use Ultralytics YOLOv7
            from ultralytics import YOLO
            
            model_name = self.config['model']['pretrained']
            self.model = YOLO(model_name)
            
            # Configure for VisDrone dataset
            self.model.model[-1].nc = self.config['data']['nc']  # Set number of classes
            self.model.model[-1].anchors = self.model.model[-1].anchors.clone()
            
            logger.info(f"Loaded YOLOv7 model: {model_name}")
            
        except ImportError:
            logger.warning("Ultralytics not available, using custom implementation")
            # Fallback to custom implementation
            self.model = YOLOv7DroneModel(
                config_path=self.config['model']['config'],
                device=str(self.device)
            )
            self.model.load_pretrained(self.config['model']['pretrained'])
    
    def setup_data(self):
        """Setup data loading for training."""
        data_config = self.config['data'].copy()
        
        # Create dataset YAML for YOLOv7
        dataset_yaml = {
            'path': data_config['path'],
            'train': data_config['train'],
            'val': data_config['val'],
            'test': data_config.get('test', data_config['val']),
            'nc': data_config['nc'],
            'names': data_config['names']
        }
        
        # Save dataset configuration
        dataset_yaml_path = self.output_dir / 'dataset.yaml'
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        logger.info(f"Dataset configuration saved: {dataset_yaml_path}")
        return str(dataset_yaml_path)
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        if hasattr(self.model, 'model'):  # Ultralytics model
            # Optimizer will be setup automatically by Ultralytics
            return
        
        # Custom optimizer setup
        opt_config = self.config['training']['optimizer']
        
        if opt_config['type'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr0'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr0'],
                momentum=opt_config['momentum'],
                weight_decay=opt_config['weight_decay']
            )
        
        # Learning rate scheduler
        scheduler_config = self.config['training']['scheduler']
        if scheduler_config['type'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif scheduler_config['type'] == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=opt_config['lrf']
            )
        
        # Mixed precision scaler
        if self.config['advanced']['amp']:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Optimizer: {opt_config['type']}, Scheduler: {scheduler_config['type']}")
    
    def train_ultralytics(self, dataset_yaml_path: str):
        """Train using Ultralytics YOLOv7."""
        train_config = self.config['training']
        val_config = self.config['validation']
        aug_config = self.config['augmentation']
        
        # Training arguments
        train_args = {
            'data': dataset_yaml_path,
            'epochs': train_config['epochs'],
            'batch': train_config['batch_size'],
            'imgsz': train_config['img_size'],
            'device': str(self.device),
            'workers': train_config['workers'],
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'exist_ok': self.config['output']['exist_ok'],
            
            # Optimizer settings
            'optimizer': train_config['optimizer']['type'],
            'lr0': train_config['optimizer']['lr0'],
            'lrf': train_config['optimizer']['lrf'],
            'momentum': train_config['optimizer']['momentum'],
            'weight_decay': train_config['optimizer']['weight_decay'],
            'warmup_epochs': train_config['optimizer']['warmup_epochs'],
            
            # Augmentation settings
            'mosaic': aug_config['mosaic'],
            'mixup': aug_config['mixup'],
            'degrees': aug_config['degrees'],
            'translate': aug_config['translate'],
            'scale': aug_config['scale'],
            'shear': aug_config['shear'],
            'perspective': aug_config['perspective'],
            'flipud': aug_config['flipud'],
            'fliplr': aug_config['fliplr'],
            'hsv_h': aug_config['hsv_h'],
            'hsv_s': aug_config['hsv_s'],
            'hsv_v': aug_config['hsv_v'],
            
            # Validation settings
            'val': True,
            'save': val_config['save_best'],
            'save_period': val_config['save_interval'],
            'conf': val_config['conf_threshold'],
            'iou': val_config['iou_threshold'],
            
            # Advanced settings
            'amp': self.config['advanced']['amp'],
            'multi_scale': self.config['advanced']['multi_scale'],
            'patience': self.config['advanced']['patience'],
            
            # Resume training if specified
            'resume': self.config['advanced']['resume'] if self.config['advanced']['resume'] else False,
        }
        
        # Add drone-specific augmentations if enabled
        if self.config['drone_specific']['small_object_aug']:
            train_args['copy_paste'] = 0.1  # Enable copy-paste for small objects
        
        logger.info("Starting Ultralytics YOLOv7 training...")
        logger.info(f"Training arguments: {train_args}")
        
        # Start training
        results = self.model.train(**train_args)
        
        logger.info("Training completed!")
        return results
    
    def train_custom(self):
        """Custom training loop (if not using Ultralytics)."""
        logger.warning("Custom training loop not implemented. Please use Ultralytics YOLOv7.")
        raise NotImplementedError("Custom training loop not implemented")
    
    def train(self):
        """Main training function."""
        logger.info("Starting YOLOv7 training for VisDrone dataset")
        
        # Setup components
        self.setup_model()
        dataset_yaml_path = self.setup_data()
        self.setup_optimizer()
        
        # Save configuration
        config_save_path = self.output_dir / 'training_config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        try:
            if hasattr(self.model, 'train'):  # Ultralytics model
                results = self.train_ultralytics(dataset_yaml_path)
            else:
                results = self.train_custom()
            
            logger.info("Training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Cleanup
            if self.config['logging']['wandb']:
                wandb.finish()
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming training from: {checkpoint_path}")
        self.config['advanced']['resume'] = checkpoint_path
        return self.train()


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train YOLOv7 on VisDrone dataset')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default='',
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device (auto, cpu, cuda, or GPU ID)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override epochs from config')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = YOLOv7Trainer(args.config)
        
        # Override config with command line arguments
        if args.device != 'auto':
            trainer.config['training']['device'] = args.device
        if args.batch_size:
            trainer.config['training']['batch_size'] = args.batch_size
        if args.epochs:
            trainer.config['training']['epochs'] = args.epochs
        if args.resume:
            trainer.config['advanced']['resume'] = args.resume
        
        # Start training
        if args.resume:
            results = trainer.resume_training(args.resume)
        else:
            results = trainer.train()
        
        print("\n✓ Training completed successfully!")
        print(f"Results saved to: {trainer.output_dir}")
        
        if hasattr(results, 'results_dict'):
            print(f"Best mAP@0.5: {results.results_dict.get('metrics/mAP_0.5', 'N/A')}")
            print(f"Best mAP@0.5:0.95: {results.results_dict.get('metrics/mAP_0.5:0.95', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()