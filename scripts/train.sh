#!/bin/bash

# Training Script for YOLOv7 + VisDrone
# This script handles the complete training pipeline

set -e  # Exit on any error

echo "=========================================="
echo "YOLOv7 + VisDrone Training Pipeline"
echo "=========================================="

# Default parameters
CONFIG_FILE="configs/training_config.yaml"
DEVICE="auto"
BATCH_SIZE=""
EPOCHS=""
RESUME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --config FILE        Training config file (default: configs/training_config.yaml)"
            echo "  --device DEVICE      Device for training (auto, cpu, cuda)"
            echo "  --batch_size SIZE    Batch size override"
            echo "  --epochs NUM         Number of epochs override"
            echo "  --resume PATH        Resume from checkpoint"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Training Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Device: $DEVICE"
if [[ -n "$BATCH_SIZE" ]]; then
    echo "  Batch size: $BATCH_SIZE"
fi
if [[ -n "$EPOCHS" ]]; then
    echo "  Epochs: $EPOCHS"
fi
if [[ -n "$RESUME" ]]; then
    echo "  Resume from: $RESUME"
fi
echo ""

# Check if virtual environment exists, create if not
if [[ ! -d "venv" ]]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [[ -f "venv/Scripts/activate" ]]; then
    # Windows
    source venv/Scripts/activate
elif [[ -f "venv/bin/activate" ]]; then
    # Unix/Linux/Mac
    source venv/bin/activate
else
    echo "Error: Could not find virtual environment activation script"
    exit 1
fi

echo "✓ Virtual environment activated"

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if data exists
if [[ ! -d "data/visdrone" ]]; then
    echo "Downloading VisDrone dataset..."
    python scripts/download_dataset.py --data_dir data/visdrone --verify
else
    echo "✓ VisDrone dataset found"
fi

# Check if YOLO format data exists
if [[ ! -d "data/yolo_format" ]]; then
    echo "Converting annotations to YOLO format..."
    python src/data_processing/visdrone_converter.py \
        --input_dir data/visdrone \
        --output_dir data/yolo_format \
        --split all
else
    echo "✓ YOLO format data found"
fi

# Build training command
TRAIN_CMD="python src/training/train_yolo.py --config $CONFIG_FILE --device $DEVICE"

if [[ -n "$BATCH_SIZE" ]]; then
    TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
fi

if [[ -n "$EPOCHS" ]]; then
    TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
fi

if [[ -n "$RESUME" ]]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
fi

echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

# Run training
eval $TRAIN_CMD

echo ""
echo "=========================================="
echo "Training completed successfully!"
echo "=========================================="

# Check for trained model
if [[ -d "results/training" ]]; then
    echo "Training results saved to: results/training/"
    
    # Find the latest model
    LATEST_MODEL=$(find results/training -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || echo "")
    
    if [[ -n "$LATEST_MODEL" ]]; then
        echo "Latest model: $LATEST_MODEL"
        echo ""
        echo "Next steps:"
        echo "1. Run inference: python src/inference/detect.py --model \"$LATEST_MODEL\" --input <image_or_directory>"
        echo "2. Evaluate model: bash scripts/evaluate.sh --model \"$LATEST_MODEL\""
    fi
fi

echo "Training pipeline completed!"