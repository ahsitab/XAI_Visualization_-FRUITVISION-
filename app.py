import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import zipfile
import io
from datetime import datetime

# Import XAI methods
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Import custom models
from models.custom_cnn import CustomCNN
from models.convnext_model import ConvNeXtTiny
from models.efficientnet_model import EfficientNetB0
from models.densenet_model import DenseNet121
from models.vgg_model import VGG16

# Page configuration
st.set_page_config(
    page_title="XAI Model Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .uploadedImage {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

class ModelManager:
    """Manages model loading and metadata"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.model_configs = {
            "ConvNeXt-Tiny": {
                "path": "ConvNeXt-Tiny_variety_classification_best(1).pth",
                "class": ConvNeXtTiny,
                "architecture": "ConvNeXt-Tiny",
                "input_size": 224,
                "num_classes": 5
            },
            "EfficientNet-B0": {
                "path": "EfficientNet-B0_variety_classification_best.pth",
                "class": EfficientNetB0,
                "architecture": "EfficientNet-B0",
                "input_size": 224,
                "num_classes": 5
            },
            "DenseNet-121": {
                "path": "DenseNet121_variety_classification_best.pth",
                "class": DenseNet121,
                "architecture": "DenseNet-121",
                "input_size": 224,
                "num_classes": 5
            },
            "VGG-16": {
                "path": "VGG16_variety_classification_best(1).pth",
                "class": VGG16,
                "architecture": "VGG-16",
                "input_size": 224,
                "num_classes": 5
            },
            "Custom-CNN": {
                "path": "custom_cnn_best.pt",
                "class": CustomCNN,
                "architecture": "Custom CNN",
                "input_size": 224,
                "num_classes": 5
            }
        }
    
    def load_model(self, model_name: str) -> Tuple[nn.Module, Dict]:
        """Load model and return model with metadata"""
        from models.model_utils import load_model_with_adaptation
        
        config = self.model_configs[model_name]
        model_path = self.models_dir / config["path"]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        model = config["class"](num_classes=config["num_classes"])
        
        try:
            # Load model with automatic state dict adaptation
            model = load_model_with_adaptation(model, model_path, device)
                
        except Exception as e:
            raise RuntimeError(f"Error loading model state dict: {str(e)}")
        
        model.to(device)
        model.eval()
        
        return model, config

class ImageProcessor:
    """Handles image preprocessing and transformations"""
    
    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], 
                               std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], 
                               std=[1., 1., 1.])
        ])
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        return self.transform(image).unsqueeze(0)
    
    def denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to displayable image"""
        img = self.inverse_transform(tensor.squeeze())
        img = img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        return img

class XAIMethods:
    """Handles all XAI method implementations"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.target_layers = self._get_target_layers()
    
    def _get_target_layers(self) -> List[nn.Module]:
        """Get target layers for CAM methods"""
        target_layers = []
        
        # Handle wrapped models (our custom implementations)
        if hasattr(self.model, 'model'):
            # This is a wrapped model, access inner model
            inner_model = self.model.model
            
            # Try to find features in inner model
            if hasattr(inner_model, 'features'):
                features = inner_model.features
                if isinstance(features, nn.Sequential):
                    # Find the last conv layer in features by searching through children
                    conv_layers = []
                    for child in features.children():
                        if isinstance(child, nn.Conv2d):
                            conv_layers.append(child)
                        else:
                            # Search within child modules
                            for module in child.modules():
                                if isinstance(module, nn.Conv2d):
                                    conv_layers.append(module)
                    
                    if conv_layers:
                        target_layers.append(conv_layers[-1])
                    else:
                        # Fallback: use the last sequential block
                        target_layers.append(features[-1])
            
            # For ConvNeXt and similar architectures, look for specific layers
            elif hasattr(inner_model, 'classifier'):
                # For ConvNeXt, use the features output
                target_layers.append(features)
        
        # Handle direct model access (DenseNet, VGG, etc.)
        elif hasattr(self.model, 'features'):
            features = self.model.features
            if isinstance(features, nn.Sequential):
                # Find last conv layer or block
                for i in range(len(features) - 1, -1, -1):
                    layer = features[i]
                    if isinstance(layer, nn.Conv2d):
                        target_layers.append(layer)
                        break
                    elif hasattr(layer, 'modules'):
                        # Search within the layer
                        for module in layer.modules():
                            if isinstance(module, nn.Conv2d):
                                target_layers.append(module)
                                break
                        if target_layers:
                            break
        
        # Specific handling for different architectures
        if not target_layers:
            # For ConvNeXt
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'features'):
                features = self.model.model.features
                # Use the last stage/block
                if len(features) > 0:
                    last_block = features[-1]
                    # Find conv layer in last block
                    for module in last_block.modules():
                        if isinstance(module, nn.Conv2d):
                            target_layers.append(module)
                            break
                    if not target_layers:
                        target_layers.append(last_block)
            
            # For DenseNet
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'features'):
                features = self.model.model.features
                # Use the denseblock output
                if len(features) > 0:
                    target_layers.append(features)
            
            # For VGG
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'features'):
                features = self.model.model.features
                # Find last conv layer
                conv_layers = []
                for module in features.modules():
                    if isinstance(module, nn.Conv2d):
                        conv_layers.append(module)
                if conv_layers:
                    target_layers.append(conv_layers[-1])
        
        # Final fallback: use the entire features as target
        if not target_layers and hasattr(self.model, 'features'):
            target_layers.append(self.model.features)
        
        # For wrapped models, use inner features
        if not target_layers and hasattr(self.model, 'model') and hasattr(self.model.model, 'features'):
            target_layers.append(self.model.model.features)
        
        return target_layers
    
    def grad_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Generate Grad-CAM heatmap"""
        if not self.target_layers:
            return np.zeros((224, 224))
        
        cam = GradCAM(model=self.model, target_layers=self.target_layers)
        targets = [ClassifierOutputTarget(target_class)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        return grayscale_cam
    
    def grad_cam_plus_plus(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Generate Grad-CAM++ heatmap"""
        if not self.target_layers:
            return np.zeros((224, 224))
        
        cam = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers)
        targets = [ClassifierOutputTarget(target_class)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        return grayscale_cam
    
    def eigen_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Generate Eigen-CAM heatmap"""
        if not self.target_layers:
            return np.zeros((224, 224))
        
        cam = EigenCAM(model=self.model, target_layers=self.target_layers)
        targets = [ClassifierOutputTarget(target_class)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        return grayscale_cam
    
    def ablation_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """Generate Ablation-CAM heatmap"""
        if not self.target_layers:
            return np.zeros((224, 224))
        
        cam = AblationCAM(model=self.model, target_layers=self.target_layers)
        targets = [ClassifierOutputTarget(target_class)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        return grayscale_cam
    
    def lime_explanation(self, image: np.ndarray, predict_fn) -> Tuple[np.ndarray, np.ndarray]:
        """Generate LIME explanation"""
        explainer = lime_image.LimeImageExplainer()
        
        explanation = explainer.explain_instance(
            image=image,
            classifier_fn=predict_fn,
            top_labels=3,
            hide_color=0,
            num_samples=1000
        )
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        
        return temp, mask

class App:
    """Main application class"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.image_processor = None
        self.xai_methods = None
        self.current_model = None
        self.current_config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Class names (update based on your dataset - fruit varieties)
        self.class_names = ['apple', 'mango', 'grape', 'banana', 'orange']
    
    def setup_sidebar(self):
        """Setup sidebar UI"""
        st.sidebar.title("🔧 Controls")
        
        # Model selection
        st.sidebar.header("Model Selection")
        selected_model = st.sidebar.selectbox(
            "Choose a model:",
            list(self.model_manager.model_configs.keys())
        )
        
        if st.sidebar.button("Load Model"):
            self.load_model(selected_model)
        
        # Image input
        st.sidebar.header("Image Input")
        input_method = st.sidebar.radio(
            "Choose input method:",
            ["Upload Image", "Sample Images"]
        )
        
        if input_method == "Upload Image":
            uploaded_file = st.sidebar.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                return Image.open(uploaded_file)
        else:
            sample_images = self.get_sample_images()
            if sample_images:
                selected_sample = st.sidebar.selectbox(
                    "Choose a sample:",
                    list(sample_images.keys())
                )
                return sample_images[selected_sample]
        
        return None
    
    def get_sample_images(self) -> Dict[str, Image.Image]:
        """Get sample images from samples directory"""
        sample_dir = Path("samples")
        samples = {}
        
        if sample_dir.exists():
            for img_path in sample_dir.glob("*.[jp][pn]g"):
                try:
                    img = Image.open(img_path).convert('RGB')
                    samples[img_path.name] = img
                except Exception as e:
                    st.error(f"Error loading sample {img_path}: {e}")
        
        return samples
    
    def load_model(self, model_name: str):
        """Load selected model"""
        with st.spinner(f"Loading {model_name}..."):
            try:
                self.current_model, self.current_config = self.model_manager.load_model(model_name)
                self.image_processor = ImageProcessor(self.current_config["input_size"])
                self.xai_methods = XAIMethods(self.current_model, self.device)
                st.success(f"✅ {model_name} loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")
    
    def predict(self, image: Image.Image) -> Tuple[torch.Tensor, List[Dict]]:
        """Make prediction on image"""
        if self.current_model is None:
            st.error("Please load a model first!")
            return None, None
        
        # Preprocess image
        input_tensor = self.image_processor.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.current_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top-3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        predictions = []
        for i in range(3):
            predictions.append({
                'class': self.class_names[top3_idx[i].item()],
                'probability': top3_prob[i].item(),
                'class_idx': top3_idx[i].item()
            })
        
        return input_tensor, predictions
    
    def generate_explanations(self, image: Image.Image, input_tensor: torch.Tensor, 
                            predictions: List[Dict]) -> Dict[str, np.ndarray]:
        """Generate all XAI explanations"""
        if self.xai_methods is None:
            return {}
        
        # Convert image for visualization
        img_array = np.array(image.resize((224, 224))) / 255.0
        
        # Get predicted class
        predicted_class = predictions[0]['class_idx']
        
        explanations = {}
        
        # Generate CAM-based explanations
        with st.spinner("Generating Grad-CAM..."):
            gradcam = self.xai_methods.grad_cam(input_tensor, predicted_class)
            explanations['Grad-CAM'] = show_cam_on_image(img_array, gradcam, use_rgb=True)
        
        with st.spinner("Generating Grad-CAM++..."):
            gradcam_pp = self.xai_methods.grad_cam_plus_plus(input_tensor, predicted_class)
            explanations['Grad-CAM++'] = show_cam_on_image(img_array, gradcam_pp, use_rgb=True)
        
        with st.spinner("Generating Eigen-CAM..."):
            eigencam = self.xai_methods.eigen_cam(input_tensor, predicted_class)
            explanations['Eigen-CAM'] = show_cam_on_image(img_array, eigencam, use_rgb=True)
        
        with st.spinner("Generating Ablation-CAM..."):
            ablationcam = self.xai_methods.ablation_cam(input_tensor, predicted_class)
            explanations['Ablation-CAM'] = show_cam_on_image(img_array, ablationcam, use_rgb=True)
        
        # Generate LIME explanation
        with st.spinner("Generating LIME explanation..."):
            def predict_fn(images):
                images = torch.tensor(images).permute(0, 3, 1, 2).float()
                images = transforms.Resize((224, 224))(images)
                images = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])(images)
                
                with torch.no_grad():
                    outputs = self.current_model(images.to(self.device))
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                return probs.cpu().numpy()
            
            try:
                lime_img, lime_mask = self.xai_methods.lime_explanation(
                    img_array, predict_fn
                )
                explanations['LIME'] = mark_boundaries(lime_img, lime_mask)
            except Exception as e:
                st.warning(f"LIME generation failed: {e}")
                explanations['LIME'] = img_array
        
        return explanations
    
    def display_results(self, image: Image.Image, predictions: List[Dict], 
                       explanations: Dict[str, np.ndarray]):
        """Display prediction results and explanations"""
        
        # Display original image and predictions
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Predictions")
            for pred in predictions:
                st.write(f"**{pred['class']}**: {pred['probability']:.2%}")
                st.progress(pred['probability'])
        
        # Display explanations
        st.subheader("🔍 Explanations")
        
        # Create grid for explanations
        cols = st.columns(3)
        
        for idx, (method_name, explanation) in enumerate(explanations.items()):
            col_idx = idx % 3
            with cols[col_idx]:
                st.write(f"**{method_name}**")
                st.image(explanation, use_column_width=True)
        
        # Download button
        if st.button("📥 Download Explanations"):
            self.download_explanations(image, explanations)
    
    def download_explanations(self, image: Image.Image, 
                            explanations: Dict[str, np.ndarray]):
        """Create downloadable zip of explanations"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add original image
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            zip_file.writestr('original_image.png', img_buffer.read())
            
            # Add explanations
            for method_name, explanation in explanations.items():
                if explanation is not None:
                    img = Image.fromarray((explanation * 255).astype(np.uint8))
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    buffer.seek(0)
                    zip_file.writestr(f'{method_name}.png', buffer.read())
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="Download All Visualizations",
            data=zip_buffer,
            file_name=f"xai_explanations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )
    
    def run(self):
        """Main application runner"""
        st.title("🔍 XAI Model Explorer")
        st.markdown("Explore trained models with various XAI techniques")
        
        # Initialize session state
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'config' not in st.session_state:
            st.session_state.config = None
        if 'image_processor' not in st.session_state:
            st.session_state.image_processor = None
        if 'xai_methods' not in st.session_state:
            st.session_state.xai_methods = None
        
        # Setup sidebar
        st.sidebar.title("🔧 Controls")
        
        # Model selection
        st.sidebar.header("Model Selection")
        selected_model = st.sidebar.selectbox(
            "Choose a model:",
            list(self.model_manager.model_configs.keys()),
            key="model_selector"
        )
        
        if st.sidebar.button("Load Model", key="load_model_btn"):
            with st.spinner(f"Loading {selected_model}..."):
                try:
                    model, config = self.model_manager.load_model(selected_model)
                    st.session_state.model = model
                    st.session_state.config = config
                    st.session_state.image_processor = ImageProcessor(config["input_size"])
                    st.session_state.xai_methods = XAIMethods(model, self.device)
                    st.success(f"✅ {selected_model} loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {e}")
        
        # Image input
        st.sidebar.header("Image Input")
        input_method = st.sidebar.radio(
            "Choose input method:",
            ["Upload Image", "Sample Images"],
            key="input_method_radio"
        )
        
        image = None
        if input_method == "Upload Image":
            uploaded_file = st.sidebar.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"],
                key="file_uploader"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
        else:
            sample_images = self.get_sample_images()
            if sample_images:
                selected_sample = st.sidebar.selectbox(
                    "Choose a sample:",
                    list(sample_images.keys()),
                    key="sample_selector"
                )
                image = sample_images[selected_sample]
        
        # Display model status
        if st.session_state.model is not None:
            st.sidebar.success("Model loaded and ready!")
        else:
            st.sidebar.warning("Please load a model first")
        
        if image is not None:
            # Display image info
            st.info(f"Image loaded: {image.size}")
            
            if st.session_state.model is not None:
                # Predict button
                if st.button("🔮 Predict & Explain", key="predict_btn"):
                    input_tensor, predictions = self.predict_with_session_state(image)
                    
                    if predictions:
                        explanations = self.generate_explanations_with_session_state(image, input_tensor, predictions)
                        self.display_results(image, predictions, explanations)
            else:
                st.warning("Please load a model before making predictions")
        
        # Footer
        st.markdown("---")
        st.markdown("Built with Streamlit, PyTorch, and various XAI libraries")
    
    def predict_with_session_state(self, image: Image.Image) -> Tuple[torch.Tensor, List[Dict]]:
        """Make prediction on image using session state"""
        if st.session_state.model is None:
            st.error("Please load a model first!")
            return None, None
        
        # Use session state components
        image_processor = st.session_state.image_processor
        model = st.session_state.model
        
        # Preprocess image
        input_tensor = image_processor.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top-3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        predictions = []
        for i in range(3):
            predictions.append({
                'class': self.class_names[top3_idx[i].item()],
                'probability': top3_prob[i].item(),
                'class_idx': top3_idx[i].item()
            })
        
        return input_tensor, predictions
    
    def generate_explanations_with_session_state(self, image: Image.Image, input_tensor: torch.Tensor, 
                            predictions: List[Dict]) -> Dict[str, np.ndarray]:
        """Generate all XAI explanations using session state"""
        if st.session_state.xai_methods is None:
            return {}
        
        # Use session state components
        xai_methods = st.session_state.xai_methods
        image_processor = st.session_state.image_processor
        
        # Convert image for visualization
        img_array = np.array(image.resize((224, 224))) / 255.0
        
        # Get predicted class
        predicted_class = predictions[0]['class_idx']
        
        explanations = {}
        
        # Generate CAM-based explanations
        with st.spinner("Generating Grad-CAM..."):
            gradcam = xai_methods.grad_cam(input_tensor, predicted_class)
            explanations['Grad-CAM'] = show_cam_on_image(img_array, gradcam, use_rgb=True)
        
        with st.spinner("Generating Grad-CAM++..."):
            gradcam_pp = xai_methods.grad_cam_plus_plus(input_tensor, predicted_class)
            explanations['Grad-CAM++'] = show_cam_on_image(img_array, gradcam_pp, use_rgb=True)
        
        with st.spinner("Generating Eigen-CAM..."):
            eigencam = xai_methods.eigen_cam(input_tensor, predicted_class)
            explanations['Eigen-CAM'] = show_cam_on_image(img_array, eigencam, use_rgb=True)
        
        with st.spinner("Generating Ablation-CAM..."):
            ablationcam = xai_methods.ablation_cam(input_tensor, predicted_class)
            explanations['Ablation-CAM'] = show_cam_on_image(img_array, ablationcam, use_rgb=True)
        
        # Generate LIME explanation
        with st.spinner("Generating LIME explanation..."):
            def predict_fn(images):
                images = torch.tensor(images).permute(0, 3, 1, 2).float()
                images = transforms.Resize((224, 224))(images)
                images = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])(images)
                
                with torch.no_grad():
                    outputs = st.session_state.model(images.to(self.device))
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                return probs.cpu().numpy()
            
            try:
                lime_img, lime_mask = xai_methods.lime_explanation(
                    img_array, predict_fn
                )
                explanations['LIME'] = mark_boundaries(lime_img, lime_mask)
            except Exception as e:
                st.warning(f"LIME generation failed: {e}")
                explanations['LIME'] = img_array
        
        return explanations

if __name__ == "__main__":
    app = App()
    app.run()
