#!/usr/bin/env python3
"""
Test script to verify model loading fixes work correctly.
This script tests loading all models with the new compatibility layer.
"""

import torch
from pathlib import Path
from models.convnext_model import ConvNeXtTiny
from models.efficientnet_model import EfficientNetB0
from models.densenet_model import DenseNet121
from models.vgg_model import VGG16
from models.model_utils import load_model_with_adaptation

def test_model_loading():
    """Test loading all models with compatibility fixes."""
    
    models_dir = Path("models")
    
    # Model configurations
    model_configs = {
        "ConvNeXt-Tiny": {
            "path": "ConvNeXt-Tiny_variety_classification_best(1).pth",
            "class": ConvNeXtTiny,
            "num_classes": 5
        },
        "EfficientNet-B0": {
            "path": "EfficientNet-B0_variety_classification_best.pth",
            "class": EfficientNetB0,
            "num_classes": 5
        },
        "DenseNet-121": {
            "path": "DenseNet121_variety_classification_best.pth",
            "class": DenseNet121,
            "num_classes": 5
        },
        "VGG-16": {
            "path": "VGG16_variety_classification_best(1).pth",
            "class": VGG16,
            "num_classes": 5
        }
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_name, config in model_configs.items():
        model_path = models_dir / config["path"]
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            continue
        
        # Initialize model
        model = config["class"](num_classes=config["num_classes"])
        
        try:
            # Load model with adaptation
            model = load_model_with_adaptation(model, model_path, device)
            print(f"{model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

if __name__ == "__main__":
    test_model_loading()
