#!/usr/bin/env python3
"""

VisDrone Dataset Download Script

This script downloads the VisDrone2019 dataset for object detection.
The dataset includes training, validation, and test sets with 10 object classes.

Usage:
    python scripts/download_dataset.py [--data_dir <path>] [--split <train|val|test|all>]
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class VisDroneDownloader:
    """VisDrone dataset downloader with progress tracking."""
    
    DATASET_URLS = {
        'train_images': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip',
        'val_images': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip',
        'test_images': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip',
    }
    
    CLASS_NAMES = [
        'ignored',      # 0 (ignored during evaluation)
        'pedestrian',   # 1
        'people',       # 2  
        'bicycle',      # 3
        'car',          # 4
        'van',          # 5
        'truck',        # 6
        'tricycle',     # 7
        'awning-tricycle', # 8
        'bus',          # 9
        'motor'         # 10
    ]
    
    def __init__(self, data_dir='data/visdrone'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_with_progress(self, url, filepath):
        """Download file with progress bar."""
        class TqdmUpTo(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filepath.name) as t:
            urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)
    
    def extract_zip(self, zip_path, extract_to):
        """Extract zip file with progress."""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
                zip_ref.extract(member, extract_to)
    
    def download_split(self, split='train'):
        """Download a specific split of the dataset."""
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        key = f'{split}_images'
        if key not in self.DATASET_URLS:
            raise ValueError(f"No URL found for split: {split}")
        
        url = self.DATASET_URLS[key]
        zip_filename = f"VisDrone2019-DET-{split}.zip"
        zip_path = self.data_dir / zip_filename
        
        print(f"Downloading {split} split...")
        
        # Download if not exists
        if not zip_path.exists():
            print(f"Downloading from: {url}")
            self.download_with_progress(url, zip_path)
        else:
            print(f"File already exists: {zip_path}")
        
        # Extract
        extract_dir = self.data_dir / split
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            print(f"Extracting to: {extract_dir}")
            self.extract_zip(zip_path, self.data_dir)
            
            # Move extracted contents to proper directory
            extracted_name = f"VisDrone2019-DET-{split}"
            if split == 'test':
                extracted_name = "VisDrone2019-DET-test-dev"
            
            extracted_path = self.data_dir / extracted_name
            if extracted_path.exists():
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)
                extracted_path.rename(extract_dir)
        else:
            print(f"Split already extracted: {extract_dir}")
        
        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()
            print(f"Cleaned up: {zip_path}")
        
        return extract_dir
    
    def download_all(self):
        """Download all splits of the dataset."""
        splits = ['train', 'val', 'test']
        downloaded_paths = {}
        
        for split in splits:
            try:
                path = self.download_split(split)
                downloaded_paths[split] = path
                print(f"✓ Successfully downloaded {split} split")
            except Exception as e:
                print(f"✗ Failed to download {split} split: {e}")
        
        return downloaded_paths
    
    def verify_dataset(self):
        """Verify dataset structure and contents."""
        print("\nVerifying dataset structure...")
        
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                print(f"✗ Missing {split} directory")
                continue
            
            # Check for images and annotations
            images_dir = split_dir / 'images'
            annotations_dir = split_dir / 'annotations'
            
            if images_dir.exists():
                image_count = len(list(images_dir.glob('*.jpg')))
                print(f"✓ {split} images: {image_count} files")
            else:
                print(f"✗ Missing {split}/images directory")
            
            if annotations_dir.exists():
                ann_count = len(list(annotations_dir.glob('*.txt')))
                print(f"✓ {split} annotations: {ann_count} files")
            else:
                print(f"✗ Missing {split}/annotations directory")
    
    def create_class_file(self):
        """Create classes.txt file for YOLO training."""
        classes_file = self.data_dir / 'classes.txt'
        with open(classes_file, 'w') as f:
            # Skip 'ignored' class (index 0)
            for class_name in self.CLASS_NAMES[1:]:
                f.write(f"{class_name}\n")
        
        print(f"✓ Created classes file: {classes_file}")
        return classes_file


def main():
    parser = argparse.ArgumentParser(description='Download VisDrone dataset')
    parser.add_argument('--data_dir', type=str, default='data/visdrone',
                       help='Directory to save dataset (default: data/visdrone)')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], 
                       default='all', help='Dataset split to download (default: all)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify dataset after download')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = VisDroneDownloader(args.data_dir)
    
    # Download requested splits
    if args.split == 'all':
        print("Downloading all VisDrone dataset splits...")
        downloader.download_all()
    else:
        print(f"Downloading {args.split} split...")
        downloader.download_split(args.split)
    
    # Create class file
    downloader.create_class_file()
    
    # Verify if requested
    if args.verify:
        downloader.verify_dataset()
    
    print("\n✓ Dataset download completed!")
    print(f"Dataset location: {os.path.abspath(args.data_dir)}")
    print("\nNext steps:")
    print("1. Convert annotations to YOLO format: python src/data_processing/visdrone_converter.py")
    print("2. Configure training: edit configs/training_config.yaml")
    print("3. Start training: bash scripts/train.sh")


if __name__ == '__main__':
    main()