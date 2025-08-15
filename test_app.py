"""
Test script to verify the XAI Model Explorer setup
"""

import os
import sys
from pathlib import Path
import torch
import streamlit as st

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        import torchvision
        import streamlit
        import PIL
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        from pytorch_grad_cam import GradCAM
        from lime import lime_image
        
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_models_directory():
    """Test if models directory exists and has files"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ Models directory not found")
        return False
    
    model_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pt"))
    
    if not model_files:
        print("❌ No model files found in models directory")
        return False
    
    print(f"✅ Found {len(model_files)} model files:")
    for f in model_files:
        print(f"  - {f.name}")
    
    return True

def test_device():
    """Test CUDA availability"""
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA not available, will use CPU")
    return True

def test_sample_images():
    """Test sample images directory"""
    samples_dir = Path("samples")
    
    if not samples_dir.exists():
        print("❌ Samples directory not found")
        return False
    
    image_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
    
    if not image_files:
        print("⚠️  No sample images found")
    else:
        print(f"✅ Found {len(image_files)} sample images")
    
    return True

def main():
    """Run all tests"""
    print("🧪 XAI Model Explorer - Test Suite")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Models Directory", test_models_directory),
        ("Device Check", test_device),
        ("Sample Images", test_sample_images)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    if all(results):
        print("✅ All tests passed! Ready to run the app.")
        print("\nRun the app with:")
        print("streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    