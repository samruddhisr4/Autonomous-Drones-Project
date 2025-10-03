"""
VisDrone to YOLO Format Converter

Converts VisDrone annotation format to YOLO format for training.
VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized)

Author: Autonomous Drone Detection Team
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class VisDroneYOLOConverter:
    """Convert VisDrone annotations to YOLO format."""
    
    # VisDrone class mapping (0 is ignored, 1-10 are valid classes)
    CLASS_MAPPING = {
        0: -1,  # ignored regions - skip
        1: 0,   # pedestrian
        2: 1,   # people
        3: 2,   # bicycle
        4: 3,   # car
        5: 4,   # van
        6: 5,   # truck
        7: 6,   # tricycle
        8: 7,   # awning-tricycle
        9: 8,   # bus
        10: 9   # motor
    }
    
    CLASS_NAMES = [
        'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    def __init__(self, visdrone_dir, output_dir, min_bbox_area=0, ignore_difficult=True):
        """
        Initialize converter.
        
        Args:
            visdrone_dir: Path to VisDrone dataset
            output_dir: Path to save YOLO format data
            min_bbox_area: Minimum bounding box area to include
            ignore_difficult: Whether to ignore difficult/occluded objects
        """
        self.visdrone_dir = Path(visdrone_dir)
        self.output_dir = Path(output_dir)
        self.min_bbox_area = min_bbox_area
        self.ignore_difficult = ignore_difficult
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """
        Convert VisDrone bbox to YOLO format.
        
        Args:
            bbox: [left, top, width, height] in pixels
            img_width: Image width
            img_height: Image height
            
        Returns:
            [x_center, y_center, width, height] normalized [0,1]
        """
        left, top, width, height = bbox
        
        # Calculate center coordinates
        x_center = left + width / 2
        y_center = top + height / 2
        
        # Normalize to [0, 1]
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        # Ensure values are in valid range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return [x_center, y_center, width, height]
    
    def parse_visdrone_annotation(self, annotation_file):
        """
        Parse VisDrone annotation file.
        
        Args:
            annotation_file: Path to .txt annotation file
            
        Returns:
            List of parsed annotations: [(class_id, bbox), ...]
        """
        annotations = []
        
        if not annotation_file.exists():
            return annotations
        
        with open(annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) != 8:
                    continue
                
                try:
                    left = int(parts[0])
                    top = int(parts[1])
                    width = int(parts[2])
                    height = int(parts[3])
                    score = int(parts[4])  # Usually 1 for ground truth
                    category = int(parts[5])
                    truncation = int(parts[6])
                    occlusion = int(parts[7])
                    
                    # Skip ignored regions
                    if category == 0:
                        continue
                    
                    # Skip difficult objects if configured
                    if self.ignore_difficult and (occlusion == 2 or truncation == 2):
                        continue
                    
                    # Skip small bounding boxes
                    if width * height < self.min_bbox_area:
                        continue
                    
                    # Map to YOLO class (subtract 1 since VisDrone classes are 1-10)
                    if category in self.CLASS_MAPPING and self.CLASS_MAPPING[category] >= 0:
                        yolo_class = self.CLASS_MAPPING[category]
                        bbox = [left, top, width, height]
                        annotations.append((yolo_class, bbox))
                
                except (ValueError, IndexError):
                    continue
        
        return annotations
    
    def convert_single_file(self, args):
        """
        Convert a single annotation file.
        
        Args:
            args: Tuple of (image_file, annotation_file, split, output_dir)
            
        Returns:
            Dict with conversion statistics
        """
        image_file, annotation_file, split, output_split_dir = args
        
        stats = {
            'processed': 0,
            'annotations': 0,
            'skipped': 0,
            'errors': 0
        }
        
        try:
            # Check if image exists
            if not image_file.exists():
                stats['skipped'] += 1
                return stats
            
            # Get image dimensions
            img = cv2.imread(str(image_file))
            if img is None:
                stats['errors'] += 1
                return stats
            
            img_height, img_width = img.shape[:2]
            
            # Parse annotations
            annotations = self.parse_visdrone_annotation(annotation_file)
            
            # Create output directories
            images_dir = output_split_dir / 'images'
            labels_dir = output_split_dir / 'labels'
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
            
            # Copy image to output directory
            output_image = images_dir / image_file.name
            if not output_image.exists():
                import shutil
                shutil.copy2(image_file, output_image)
            
            # Convert and save annotations
            output_label = labels_dir / (image_file.stem + '.txt')
            with open(output_label, 'w') as f:
                for class_id, bbox in annotations:
                    yolo_bbox = self.convert_bbox_to_yolo(bbox, img_width, img_height)
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
            
            stats['processed'] += 1
            stats['annotations'] += len(annotations)
            
        except Exception as e:
            stats['errors'] += 1
            print(f"Error processing {image_file}: {e}")
        
        return stats
    
    def convert_split(self, split='train', num_workers=None):
        """
        Convert a dataset split to YOLO format.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            num_workers: Number of worker processes (None for auto)
        """
        if num_workers is None:
            num_workers = min(8, mp.cpu_count())
        
        split_dir = self.visdrone_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        images_dir = split_dir / 'images'
        annotations_dir = split_dir / 'annotations'
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # Get all image files
        image_files = list(images_dir.glob('*.jpg'))
        if not image_files:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"Converting {split} split: {len(image_files)} images")
        
        # Create output directory for this split
        output_split_dir = self.output_dir / split
        output_split_dir.mkdir(exist_ok=True)
        
        # Prepare arguments for multiprocessing
        convert_args = []
        for image_file in image_files:
            annotation_file = annotations_dir / (image_file.stem + '.txt')
            convert_args.append((image_file, annotation_file, split, output_split_dir))
        
        # Process files
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(self.convert_single_file, convert_args),
                    total=len(convert_args),
                    desc=f"Converting {split}"
                ))
        else:
            results = []
            for args in tqdm(convert_args, desc=f"Converting {split}"):
                results.append(self.convert_single_file(args))
        
        # Aggregate statistics
        total_stats = {
            'processed': sum(r['processed'] for r in results),
            'annotations': sum(r['annotations'] for r in results),
            'skipped': sum(r['skipped'] for r in results),
            'errors': sum(r['errors'] for r in results)
        }
        
        print(f"✓ {split} conversion completed:")
        print(f"  - Processed: {total_stats['processed']} images")
        print(f"  - Annotations: {total_stats['annotations']} objects")
        print(f"  - Skipped: {total_stats['skipped']} files")
        print(f"  - Errors: {total_stats['errors']} files")
        
        return total_stats
    
    def create_dataset_yaml(self):
        """Create dataset YAML file for YOLO training."""
        yaml_content = f"""# VisDrone dataset configuration for YOLO training
path: {self.output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: {len(self.CLASS_NAMES)}

# Class names
names: {self.CLASS_NAMES}
"""
        
        yaml_file = self.output_dir / 'dataset.yaml'
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"✓ Created dataset YAML: {yaml_file}")
        return yaml_file
    
    def convert_all(self, num_workers=None):
        """Convert all dataset splits."""
        splits = ['train', 'val', 'test']
        total_stats = {}
        
        for split in splits:
            try:
                stats = self.convert_split(split, num_workers)
                total_stats[split] = stats
            except Exception as e:
                print(f"✗ Failed to convert {split}: {e}")
                total_stats[split] = None
        
        # Create dataset YAML
        self.create_dataset_yaml()
        
        return total_stats


def main():
    parser = argparse.ArgumentParser(description='Convert VisDrone annotations to YOLO format')
    parser.add_argument('--input_dir', type=str, default='data/visdrone',
                       help='Input VisDrone dataset directory')
    parser.add_argument('--output_dir', type=str, default='data/yolo_format',
                       help='Output directory for YOLO format')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'],
                       default='all', help='Split to convert')
    parser.add_argument('--min_bbox_area', type=int, default=0,
                       help='Minimum bounding box area (pixels)')
    parser.add_argument('--include_difficult', action='store_true',
                       help='Include difficult/occluded objects')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Create converter
    converter = VisDroneYOLOConverter(
        visdrone_dir=args.input_dir,
        output_dir=args.output_dir,
        min_bbox_area=args.min_bbox_area,
        ignore_difficult=not args.include_difficult
    )
    
    # Convert data
    if args.split == 'all':
        print("Converting all splits to YOLO format...")
        converter.convert_all(args.workers)
    else:
        print(f"Converting {args.split} split to YOLO format...")
        converter.convert_split(args.split, args.workers)
    
    print("\n✓ Conversion completed!")
    print(f"YOLO format dataset saved to: {os.path.abspath(args.output_dir)}")
    print("\nNext steps:")
    print("1. Review dataset.yaml configuration")
    print("2. Start training: python src/training/train_yolo.py")


if __name__ == '__main__':
    main()