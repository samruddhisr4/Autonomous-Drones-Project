#!/bin/bash

echo "==============================================="
echo "🚁 Autonomous Drone Detection - Linux/Mac Setup"
echo "==============================================="
echo

# Check Python
echo "📋 Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✅ Python3 found"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "✅ Python found"
else
    echo "❌ Python not found! Please install Python 3.7+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
echo
echo "🔧 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "✅ Virtual environment already exists"
else
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
fi

# Activate virtual environment and install dependencies
echo
echo "📦 Activating virtual environment and installing dependencies..."
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install from requirements.txt"
        echo "Trying essential packages only..."
        pip install torch torchvision opencv-python ultralytics matplotlib plotly pandas numpy seaborn scikit-learn pyyaml
    fi
else
    echo "Installing essential packages..."
    pip install torch torchvision opencv-python ultralytics matplotlib plotly pandas numpy seaborn scikit-learn pyyaml
fi

# Create directories
echo
echo "📁 Creating directories..."
mkdir -p data/visdrone data/yolo_format results/models results/comprehensive_analysis sample_images
echo "✅ Directories created"

# Run quick setup
echo
echo "🚀 Running quick setup..."
python quick_setup.py

echo
echo "==============================================="
echo "🎉 SETUP COMPLETE!"
echo "==============================================="
echo
echo "🎯 NEXT STEPS:"
echo "1. Add test images to sample_images/ folder"
echo "2. Run: python src/inference/detect.py --input sample_images --output results/detections --device cpu"
echo "3. Generate analysis: python create_results_summary.py"
echo "4. View results: Open results/comprehensive_analysis/detection_performance_summary.html"
echo
echo "💡 Remember to activate virtual environment: source venv/bin/activate"
echo