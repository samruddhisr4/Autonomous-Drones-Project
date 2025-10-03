@echo off
echo ===============================================
echo 🚁 Autonomous Drone Detection - Windows Setup
echo ===============================================
echo.

echo 📋 Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.7+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✅ Python found

echo.
echo 🔧 Creating virtual environment...
if exist venv (
    echo ✅ Virtual environment already exists
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
)

echo.
echo 📦 Activating virtual environment and installing dependencies...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    echo Trying essential packages only...
    pip install torch torchvision opencv-python ultralytics matplotlib plotly pandas numpy seaborn scikit-learn pyyaml
)

echo.
echo 📁 Creating directories...
mkdir data\visdrone 2>nul
mkdir data\yolo_format 2>nul
mkdir results\models 2>nul
mkdir results\comprehensive_analysis 2>nul
mkdir sample_images 2>nul
echo ✅ Directories created

echo.
echo 🚀 Running quick setup...
python quick_setup.py

echo.
echo ===============================================
echo 🎉 SETUP COMPLETE!
echo ===============================================
echo.
echo 🎯 NEXT STEPS:
echo 1. Add test images to sample_images\ folder
echo 2. Run: python src\inference\detect.py --input sample_images --output results\detections --device cpu
echo 3. Generate analysis: python create_results_summary.py
echo 4. View results: Open results\comprehensive_analysis\detection_performance_summary.html
echo.
echo 💡 Remember to activate virtual environment: venv\Scripts\activate
echo.
pause