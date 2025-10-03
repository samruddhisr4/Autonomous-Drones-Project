# 🚁 Complete Setup Guide - Autonomous Drone Detection

**For running the project on a new laptop from scratch**

## 📋 Prerequisites

### System Requirements:

- **Python 3.7+** (Python 3.8-3.11 recommended)
- **Git** installed
- **8GB+ RAM** (16GB recommended)
- **10GB+ free disk space**
- **Windows/Linux/macOS** supported

### Check Prerequisites:

```bash
# Check Python version
python --version
# or
python3 --version

# Check Git
git --version

# Check available disk space
dir  # Windows
ls -la  # Linux/Mac
```

---

## 🚀 Step-by-Step Setup Instructions

### **Step 1: Clone the Repository**

```bash
# Clone the project
git clone https://github.com/samruddhisr4/Autonomous-Drones-Project.git

# Navigate to project directory
cd Autonomous-Drones-Project
```

### **Step 2: Create Virtual Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Verify activation (should show (venv) prefix)
# Your prompt should look like: (venv) C:\path\to\project>
```

### **Step 3: Install Dependencies**

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

### **Step 4: Project Setup**

```bash
# Install project in development mode
pip install -e .

# Create necessary directories
mkdir -p data/visdrone data/yolo_format results/models
```

---

## 📊 Quick Demo (No Dataset Required)

### **Option A: Run with Sample Images**

```bash
# Create a sample images directory
mkdir sample_images

# Download sample drone images (or copy your own)
# Place some .jpg images in sample_images/

# Run detection with pre-trained model
python src/inference/detect.py --input sample_images --output results/quick_test --device cpu
```

---

## 🎯 Full Pipeline (With Dataset)

### **Step 5: Download Dataset (Optional)**

```bash
# Download VisDrone dataset subset (automated)
python scripts/download_dataset.py --data_dir data/visdrone --subset

# OR manually download from:
# https://github.com/VisDrone/VisDrone-Dataset
# Extract to data/visdrone/
```

### **Step 6: Convert Dataset**

```bash
# Convert VisDrone to YOLO format
python src/data_processing/visdrone_converter.py \
    --input_dir data/visdrone \
    --output_dir data/yolo_format \
    --split all
```

### **Step 7: Quick Training (Optional)**

```bash
# Quick training with CPU (3 epochs for demo)
yolo train data=data/yolo_format/dataset.yaml model=yolov8n.pt epochs=3 device=cpu

# OR use the automated pipeline
python full_dataset_pipeline.py --mode train --epochs 3 --device cpu
```

### **Step 8: Run Detection Pipeline**

```bash
# Run detection with Betti filtering
python src/inference/detect.py \
    --model runs/detect/train/weights/best.pt \
    --input data/yolo_format/val/images \
    --output results/detections \
    --device cpu \
    --save-vis

# OR use quick demo
python quick_results_pipeline.py
```

### **Step 9: Generate Analysis & Results**

```bash
# Generate comprehensive analysis
python create_results_summary.py

# Generate interactive dashboards
python create_comprehensive_analysis.py
```

---

## 📊 View Results

### **Method 1: Command Line Results**

```bash
# Check detection outputs
ls results/detections/

# View JSON results
python -m json.tool results/quick_demo/sample_result.json
```

### **Method 2: Interactive Dashboards**

```bash
# Open HTML dashboards in browser
# Windows:
start results/comprehensive_analysis/detection_performance_summary.html

# Linux:
xdg-open results/comprehensive_analysis/detection_performance_summary.html

# Mac:
open results/comprehensive_analysis/detection_performance_summary.html
```

### **Method 3: View Charts**

```bash
# Check generated charts
ls results/comprehensive_analysis/
# Files include:
# - performance_analysis_charts.png
# - detection_performance_summary.html
# - performance_comparison.csv
```

---

## 🔧 Troubleshooting

### **Common Issues & Solutions:**

#### **1. Python Version Issues:**

```bash
# If python command doesn't work, try:
python3 --version
# Use python3 instead of python for all commands
```

#### **2. Permission Issues (Linux/Mac):**

```bash
# If permission denied:
chmod +x scripts/*.sh
sudo pip install -r requirements.txt
```

#### **3. CUDA Not Available:**

```bash
# Always use CPU for compatibility:
python src/inference/detect.py --device cpu
```

#### **4. Memory Issues:**

```bash
# Reduce batch size in configs/training_config.yaml:
# batch_size: 4  # instead of 16
```

#### **5. Missing Dependencies:**

```bash
# Install missing packages:
pip install torch torchvision opencv-python matplotlib plotly pandas numpy
```

#### **6. Dataset Download Issues:**

```bash
# Skip dataset download and use sample images:
mkdir sample_images
# Add your own images to sample_images/
```

---

## ⚡ Quick Test Commands

### **Minimal Working Example:**

```bash
# 1. Setup
git clone https://github.com/samruddhisr4/Autonomous-Drones-Project.git
cd Autonomous-Drones-Project
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Quick test
mkdir test_images
# Add some .jpg images to test_images/
python src/inference/detect.py --input test_images --output results/test --device cpu

# 3. Generate analysis
python create_results_summary.py

# 4. View results
start results/comprehensive_analysis/detection_performance_summary.html  # Windows
```

---

## 📈 Expected Results

After successful execution, you should see:

### **File Structure:**

```
results/
├── comprehensive_analysis/
│   ├── detection_performance_summary.html  ← Main dashboard
│   ├── performance_analysis_charts.png     ← Key visualizations
│   ├── performance_comparison.csv          ← Metrics table
│   └── detection_analysis_charts.html      ← Interactive charts
├── detections/                             ← Detection outputs
└── quick_demo/                             ← Sample results
```

### **Key Performance Metrics:**

- **28-45% False Positive Reduction**
- **5.6-6.9 FPS Processing Speed**
- **<5ms Filtering Overhead**
- **High Detection Accuracy**

### **Generated Files:**

1. **Interactive Dashboards** (HTML files)
2. **Performance Charts** (PNG files)
3. **Data Tables** (CSV files)
4. **Detection Visualizations** (comparison images)

---

## 🎯 Success Checklist

- [ ] ✅ Virtual environment created and activated
- [ ] ✅ Dependencies installed successfully
- [ ] ✅ Project directory structure created
- [ ] ✅ Detection pipeline runs without errors
- [ ] ✅ Results generated in results/comprehensive_analysis/
- [ ] ✅ HTML dashboards open in browser
- [ ] ✅ Performance metrics displayed correctly

---

## 💡 Tips for Your Friend

1. **Start Small**: Begin with the quick demo using sample images
2. **Check Each Step**: Verify each command completes successfully
3. **Use CPU Mode**: Always add `--device cpu` for compatibility
4. **Save Progress**: Take screenshots of successful results
5. **Ask for Help**: If stuck, share the exact error message

---

## 📞 Support

If your friend encounters issues:

1. **Check the error message** and compare with troubleshooting section
2. **Verify Python and Git versions** meet requirements
3. **Ensure virtual environment is activated** (shows (venv) prefix)
4. **Try the minimal working example** first
5. **Share specific error messages** for debugging

**🎉 Following these steps will give your friend a complete working autonomous drone detection system with comprehensive analysis and results!**
