# 🍎 XAI Fruit Vision - Streamlit Explainable AI App

A comprehensive Streamlit application that provides explainable AI (XAI) visualizations for fruit classification models. This project demonstrates five different explanation techniques for CNN model predictions.

## 🎯 Project Overview

This application allows users to:
- Select from multiple pre-trained CNN models
- Upload fruit images or use sample images
- View predictions with confidence scores
- Generate five different XAI explanations for the same prediction
- Export visualizations for further analysis


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository**
```bash
git clone [your-repo-url]
cd XAI_V(Streamlit)
```

2. **Create virtual environment**
```bash
python -m venv xai_env
source xai_env/bin/activate  # On Windows: xai_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download model weights**
   - Download weights from [Google Drive Link]
   - Place weights in `models/weights/` directory
   - Ensure the folder structure matches:
   ```
   models/
   ├── weights/
   │   ├── efficientnet_b0_fruit91.pth
   │   ├── densenet121_fruit91.pth
   │   ├── convnext_tiny_fruit91.pth
   │   ├── vgg16_fruit91.pth
   │   └── custom_cnn_fruit91.pth
   ```

5. **Run the application**
```bash
streamlit run app.py
```

## 📊 Available Models

| Model | Architecture | Input Size | Classes | Accuracy |
|-------|--------------|------------|---------|----------|
| EfficientNet-B0 | CNN | 224x224 | 91 | 98.2% |
| DenseNet-121 | CNN | 224x224 | 91 | 97.8% |
| ConvNeXt-Tiny | CNN | 224x224 | 91 | 98.5% |
| VGG-16 | CNN | 224x224 | 91 | 96.9% |
| Custom CNN | CNN | 224x224 | 91 | 95.4% |

## 🔍 XAI Techniques Implemented

### 1. **Grad-CAM**
- Highlights important regions for the predicted class
- Uses gradient information from the final convolutional layer

### 2. **Grad-CAM++**
- Improved version of Grad-CAM
- Better localization of multiple occurrences of the same class

### 3. **Eigen-CAM**
- Uses principal components of activations
- Provides a more holistic view of model attention

### 4. **Ablation-CAM**
- Systematically removes parts of the image
- Shows which regions most affect the prediction

### 5. **LIME (Local Interpretable Model-agnostic Explanations)**
- Perturbs the image and observes prediction changes
- Creates interpretable superpixel-based explanations

## 🖥️ Usage Guide

### 1. Model Selection
- Use the sidebar dropdown to select your preferred model
- View model metadata including architecture details and performance metrics

### 2. Image Input
- **Upload**: Drag & drop or browse for JPG/PNG images
- **Sample**: Choose from pre-loaded fruit images in the samples folder

### 3. Prediction & Explanations
- View top-3 predictions with confidence scores
- Generate all five XAI explanations simultaneously
- Compare different explanation techniques side-by-side

### 4. Export Results
- Download individual explanations as PNG images
- Export complete analysis as ZIP file containing all visualizations

## 📁 Project Structure

```
XAI_V(Streamlit)/
├── app.py                 # Main Streamlit application
├── models/                # Model definitions and utilities
│   ├── __init__.py
│   ├── custom_cnn.py
│   ├── densenet_model.py
│   ├── efficientnet_model.py
│   ├── convnext_model.py
│   ├── vgg_model.py
│   └── model_utils.py
├── samples/               # Sample fruit images
├── xai_env/              # Virtual environment
├── requirements.txt      # Python dependencies
├── setup.py             # Setup script
├── test_app.py          # Unit tests
└── README.md            # This file
```

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python test_app.py
python test_model_loading.py
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce batch size in model loading
- Use CPU mode: `export CUDA_VISIBLE_DEVICES=""`

**2. Model weights not found**
- Verify weights are in `models/weights/` directory
- Check file names match expected format

**3. Streamlit not launching**
- Ensure virtual environment is activated
- Check port 8501 is available: `streamlit run app.py --server.port 8502`

## 📸 Demo Screenshots

### 1. Model Selection Interface
![Model Selection](./screenshots/model_selection.png)

### 2. Image Upload & Prediction
![Prediction Results](./screenshots/prediction_results.png)

### 3. XAI Explanations Grid
![XAI Explanations](./screenshots/xai_explanations.png)

*Note: Add actual screenshots to a `screenshots/` folder*

## 🎯 Demo Checklist

- [ ] App launches successfully
- [ ] Model selection dropdown populated
- [ ] Model metadata displayed correctly
- [ ] Image upload functionality works
- [ ] Top-3 predictions shown with probabilities
- [ ] All 5 XAI explanations generated
- [ ] Explanations clearly labeled
- [ ] Side-by-side comparison view
- [ ] Export functionality available

## 📚 References

- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [Grad-CAM++: Improved Visual Explanations](https://arxiv.org/abs/1710.11063)
- [Eigen-CAM: Class Activation Map using Principal Components](https://arxiv.org/abs/2008.00299)
- [LIME: Local Interpretable Model-agnostic Explanations](https://arxiv.org/abs/1606.05386)

## 📄 License

This project is created for educational purposes as part of CSE-366 course requirements.

## 🤝 Contributing

For issues or improvements, please create an issue or submit a pull request.

---

**Submission Date**: 17 August 2025  
**Course**: CSE-366 - Explainable AI  
**Instructor**: MRAR
