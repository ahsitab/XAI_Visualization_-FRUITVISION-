#!/usr/bin/env python3
"""
Debug version of the app to identify why predictions/explanations aren't showing
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import traceback

# Import custom models
from models.custom_cnn import CustomCNN
from models.convnext_model import ConvNeXtTiny
from models.efficientnet_model import EfficientNetB0
from models.densenet_model import DenseNet121
from models.vgg_model import VGG16
from models.model_utils import load_model_with_adaptation

class DebugApp:
    def __init__(self):
        self.models_dir = Path("models")
        self.model_configs = {
            "ConvNeXt-Tiny": {
                "path": "ConvNeXt-Tiny_variety_classification_best(1).pth",
                "class": ConvNeXtTiny,
                "input_size": 224,
                "num_classes": 5
            },
            "EfficientNet-B0": {
                "path": "EfficientNet-B0_variety_classification_best.pth",
                "class": EfficientNetB0,
                "input_size": 224,
                "num_classes": 5
            },
            "DenseNet-121": {
                "path": "DenseNet121_variety_classification_best.pth",
                "class": DenseNet121,
                "input_size": 224,
                "num_classes": 5
            },
            "VGG-16": {
                "path": "VGG16_variety_classification_best(1).pth",
                "class": VGG16,
                "input_size": 224,
                "num_classes": 5
            }
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_name):
        """Load model with debug info"""
        config = self.model_configs[model_name]
        model_path = self.models_dir / config["path"]
        
        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            return None
        
        model = config["class"](num_classes=config["num_classes"])
        
        try:
            model = load_model_with_adaptation(model, model_path, self.device)
            model.to(self.device)
            model.eval()
            
            # Debug: Print model structure
            st.write("Model loaded successfully!")
            st.write(f"Model type: {type(model)}")
            st.write(f"Model device: {next(model.parameters()).device}")
            
            # Debug: Check model structure
            st.write("Model structure:")
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    st.write(f"  {name}: {type(module)}")
            
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.error(traceback.format_exc())
            return None

    def debug_prediction(self, model, image):
        """Debug prediction process"""
        try:
            # Preprocess
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            st.write(f"Input tensor shape: {input_tensor.shape}")
            st.write(f"Input tensor device: {input_tensor.device}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_tensor)
                st.write(f"Model output shape: {outputs.shape}")
                st.write(f"Model output: {outputs}")
                
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                st.write(f"Probabilities: {probabilities}")
                
                top3_prob, top3_idx = torch.topk(probabilities, 3)
                
                predictions = []
                for i in range(3):
                    predictions.append({
                        'class': self.class_names[top3_idx[i].item()],
                        'probability': top3_prob[i].item(),
                        'class_idx': top3_idx[i].item()
                    })
                
                return input_tensor, predictions
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.error(traceback.format_exc())
            return None, None

    def debug_target_layers(self, model):
        """Debug target layer detection"""
        target_layers = []
        
        st.write("=== Debugging Target Layers ===")
        
        # Check model structure
        st.write("Model attributes:")
        for attr in dir(model):
            if not attr.startswith('_'):
                st.write(f"  {attr}: {type(getattr(model, attr))}")
        
        # Try different approaches
        if hasattr(model, 'model'):
            st.write("Found wrapped model structure")
            inner_model = model.model
            if hasattr(inner_model, 'features'):
                features = inner_model.features
                st.write(f"Features type: {type(features)}")
                if isinstance(features, nn.Sequential):
                    st.write(f"Features length: {len(features)}")
                    if len(features) > 0:
                        target_layers.append(features[-1])
                        st.write(f"Added last features layer: {type(features[-1])}")
        
        elif hasattr(model, 'features'):
            st.write("Found direct features")
            features = model.features
            st.write(f"Features type: {type(features)}")
            if isinstance(features, nn.Sequential):
                st.write(f"Features length: {len(features)}")
                if len(features) > 0:
                    target_layers.append(features[-1])
                    st.write(f"Added last features layer: {type(features[-1])}")
        
        # Fallback to any conv layer
        if not target_layers:
            st.write("No target layers found, searching for conv layers...")
            conv_layers = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    conv_layers.append((name, module))
            if conv_layers:
                st.write(f"Found {len(conv_layers)} conv layers")
                for name, layer in conv_layers[-3:]:  # Show last 3
                    st.write(f"  {name}: {layer}")
                target_layers.append(conv_layers[-1][1])
        
        st.write(f"Final target layers: {len(target_layers)}")
        return target_layers

    def run(self):
        st.title("🔍 Debug XAI Model Explorer")
        
        # Model selection
        model_name = st.selectbox("Choose a model:", list(self.model_configs.keys()))
        
        if st.button("Load Model"):
            model = self.load_model(model_name)
            if model:
                st.session_state['model'] = model
        
        # Image input
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Debug Predict"):
                if 'model' in st.session_state:
                    model = st.session_state['model']
                    
                    # Debug prediction
                    input_tensor, predictions = self.debug_prediction(model, image)
                    
                    if predictions:
                        st.write("=== Predictions ===")
                        for pred in predictions:
                            st.write(f"{pred['class']}: {pred['probability']:.2%}")
                        
                        # Debug target layers
                        target_layers = self.debug_target_layers(model)
                        
                        if target_layers:
                            st.write("=== Target Layers Found ===")
                            st.write(f"Number of target layers: {len(target_layers)}")
                            for i, layer in enumerate(target_layers):
                                st.write(f"Layer {i}: {type(layer)}")
                        else:
                            st.error("No target layers found for XAI methods!")
                else:
                    st.error("Please load a model first!")

if __name__ == "__main__":
    debug_app = DebugApp()
    debug_app.run()
