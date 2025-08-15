"""
XAI Model Explorer - Setup Script
================================

This script helps set up the environment and verify all dependencies for the XAI Model Explorer.

Usage:
    python setup.py
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print("✅ Python version:", sys.version)
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def verify_models():
    """Verify model files exist"""
    models_dir = Path("models")
    required_models = [
        "ConvNeXt-Tiny_variety_classification_best(1).pth",
        "custom_cnn_best.pt",
        "DenseNet121_variety_classification_best.pth",
        "EfficientNet-B0_variety_classification_best.pth",
        "VGG16_variety_classification_best(1).pth"
    ]
    
    print("\n🔍 Checking model files...")
    missing_models = []
    
    for model_file in required_models:
        model_path = models_dir / model_file
        if model_path.exists():
            print(f"✅ {model_file}")
        else:
            print(f"❌ {model_file} - Missing")
            missing_models.append(model_file)
    
    if missing_models:
        print(f"\n⚠️  Missing models: {missing_models}")
        print("Please ensure all model files are placed in the 'models' directory")
        return False
    
    print("✅ All model files verified")
    return True

def create_sample_images():
    """Create sample images directory with placeholder images"""
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    print("\n🖼️  Setting up sample images...")
    
    # Create a simple placeholder image
    try:
        from PIL import Image
        import numpy as np
        
        # Create sample images with different patterns
        for i, name in enumerate(['sample1.jpg', 'sample2.jpg', 'sample3.jpg']):
            # Create a simple pattern
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(samples_dir / name)
            print(f"✅ Created {name}")
        
        return True
    except ImportError:
        print("⚠️  PIL not available, skipping sample image creation")
        return True

def main():
    """Main setup function"""
    print("🚀 XAI Model Explorer Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Failed to install requirements. Please install manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Verify models
    if not verify_models():
        print("\n⚠️  Some models are missing. Please add them to the 'models' directory.")
    
    # Create sample images
    create_sample_images()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\nTo run the application:")
    print("streamlit run app.py")
    print("\nOr use:")
    print("python -m streamlit run app.py")

if __name__ == "__main__":
    main()
