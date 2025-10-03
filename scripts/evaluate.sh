#!/bin/bash

# Evaluation Script for YOLOv7 + Betti Filtering
# This script runs comprehensive evaluation and comparison

set -e  # Exit on any error

echo "=========================================="
echo "YOLOv7 + Betti Filtering Evaluation"
echo "=========================================="

# Default parameters
MODEL_PATH=""
TEST_DATA="data/yolo_format/test"
OUTPUT_DIR="results/evaluation"
CREATE_VISUALIZATIONS="true"
RUN_BASELINE="true"
RUN_FILTERED="true"
CONF_THRESHOLD="0.25"
IOU_THRESHOLD="0.45"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --test_data)
            TEST_DATA="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --conf)
            CONF_THRESHOLD="$2"
            shift 2
            ;;
        --iou)
            IOU_THRESHOLD="$2"
            shift 2
            ;;
        --no-vis)
            CREATE_VISUALIZATIONS="false"
            shift
            ;;
        --baseline-only)
            RUN_FILTERED="false"
            shift
            ;;
        --filtered-only)
            RUN_BASELINE="false"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --model MODEL_PATH [options]"
            echo "Options:"
            echo "  --model PATH         Path to trained model (required)"
            echo "  --test_data PATH     Path to test data (default: data/yolo_format/test)"
            echo "  --output DIR         Output directory (default: results/evaluation)"
            echo "  --conf THRESHOLD     Confidence threshold (default: 0.25)"
            echo "  --iou THRESHOLD      IoU threshold (default: 0.45)"
            echo "  --no-vis            Skip visualization creation"
            echo "  --baseline-only     Run only baseline evaluation"
            echo "  --filtered-only     Run only filtered evaluation"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required parameters
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: Model path is required. Use --model PATH"
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "Evaluation Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Test data: $TEST_DATA"
echo "  Output directory: $OUTPUT_DIR"
echo "  Confidence threshold: $CONF_THRESHOLD"
echo "  IoU threshold: $IOU_THRESHOLD"
echo "  Create visualizations: $CREATE_VISUALIZATIONS"
echo "  Run baseline: $RUN_BASELINE"
echo "  Run filtered: $RUN_FILTERED"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment
if [[ -f "venv/Scripts/activate" ]]; then
    source venv/Scripts/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run train.sh first."
    exit 1
fi

echo "✓ Virtual environment activated"

# Check test data
if [[ ! -d "$TEST_DATA" ]]; then
    echo "Error: Test data directory not found: $TEST_DATA"
    echo "Make sure you have converted the dataset to YOLO format."
    exit 1
fi

# Find test images
TEST_IMAGES_DIR="$TEST_DATA/images"
if [[ ! -d "$TEST_IMAGES_DIR" ]]; then
    echo "Error: Test images directory not found: $TEST_IMAGES_DIR"
    exit 1
fi

NUM_TEST_IMAGES=$(find "$TEST_IMAGES_DIR" -name "*.jpg" -o -name "*.png" | wc -l)
echo "Found $NUM_TEST_IMAGES test images"

if [[ $NUM_TEST_IMAGES -eq 0 ]]; then
    echo "Error: No test images found in $TEST_IMAGES_DIR"
    exit 1
fi

# Run baseline evaluation (YOLOv7 only)
if [[ "$RUN_BASELINE" == "true" ]]; then
    echo ""
    echo "Running baseline evaluation (YOLOv7 only)..."
    
    BASELINE_OUTPUT="$OUTPUT_DIR/baseline"
    mkdir -p "$BASELINE_OUTPUT"
    
    python src/inference/detect.py \
        --model "$MODEL_PATH" \
        --input "$TEST_IMAGES_DIR" \
        --output "$BASELINE_OUTPUT" \
        --conf "$CONF_THRESHOLD" \
        --iou "$IOU_THRESHOLD" \
        --no-betti \
        --save-vis
    
    echo "✓ Baseline evaluation completed"
fi

# Run filtered evaluation (YOLOv7 + Betti filtering)
if [[ "$RUN_FILTERED" == "true" ]]; then
    echo ""
    echo "Running filtered evaluation (YOLOv7 + Betti filtering)..."
    
    FILTERED_OUTPUT="$OUTPUT_DIR/filtered"
    mkdir -p "$FILTERED_OUTPUT"
    
    python src/inference/detect.py \
        --model "$MODEL_PATH" \
        --input "$TEST_IMAGES_DIR" \
        --output "$FILTERED_OUTPUT" \
        --conf "$CONF_THRESHOLD" \
        --iou "$IOU_THRESHOLD" \
        --filter-type "standard" \
        --save-vis
    
    echo "✓ Filtered evaluation completed"
fi

# Compare results and compute metrics
if [[ "$RUN_BASELINE" == "true" && "$RUN_FILTERED" == "true" ]]; then
    echo ""
    echo "Computing evaluation metrics..."
    
    # Check if results exist
    BASELINE_RESULTS="$OUTPUT_DIR/baseline/batch_summary.json"
    FILTERED_RESULTS="$OUTPUT_DIR/filtered/batch_summary.json"
    
    if [[ -f "$BASELINE_RESULTS" && -f "$FILTERED_RESULTS" ]]; then
        python src/evaluation/metrics.py \
            --baseline "$BASELINE_RESULTS" \
            --filtered "$FILTERED_RESULTS" \
            --output "$OUTPUT_DIR" \
            --save-plots
        
        echo "✓ Evaluation metrics computed"
    else
        echo "Warning: Could not find result files for metric computation"
    fi
fi

# Create visualizations
if [[ "$CREATE_VISUALIZATIONS" == "true" ]]; then
    echo ""
    echo "Creating visualizations..."
    
    python src/evaluation/visualize.py \
        --results_dir "$OUTPUT_DIR" \
        --output_dir "$OUTPUT_DIR/visualizations"
    
    echo "✓ Visualizations created"
fi

echo ""
echo "=========================================="
echo "Evaluation completed successfully!"
echo "=========================================="

# Print summary
echo "Results saved to: $OUTPUT_DIR"
echo ""

if [[ -f "$OUTPUT_DIR/evaluation_report.json" ]]; then
    echo "Evaluation report: $OUTPUT_DIR/evaluation_report.json"
fi

if [[ -d "$OUTPUT_DIR/visualizations" ]]; then
    echo "Visualizations: $OUTPUT_DIR/visualizations/"
fi

if [[ "$RUN_BASELINE" == "true" ]]; then
    echo "Baseline results: $OUTPUT_DIR/baseline/"
fi

if [[ "$RUN_FILTERED" == "true" ]]; then
    echo "Filtered results: $OUTPUT_DIR/filtered/"
fi

echo ""
echo "Key files to review:"
echo "  - evaluation_report.json: Detailed metrics and comparisons"
echo "  - visualizations/summary_visualization.png: Main results summary"
if [[ "$RUN_BASELINE" == "true" && "$RUN_FILTERED" == "true" ]]; then
    echo "  - Compare baseline/ and filtered/ directories for detailed results"
fi

echo ""
echo "Evaluation pipeline completed!"