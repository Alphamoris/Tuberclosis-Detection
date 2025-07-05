# 🫁 Tuberculosis Detection System

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue" alt="Python 3.7+">
  <img src="https://img.shields.io/badge/TensorFlow-2.0%2B-orange" alt="TensorFlow 2.0+">
  <img src="https://img.shields.io/badge/Streamlit-1.0%2B-red" alt="Streamlit 1.0+">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT">
</div>

<p align="center">
  <i>An advanced deep learning system for detecting tuberculosis from chest X-ray images.</i>
</p>

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Setup Instructions](#setup-instructions)
- [Training Models](#training-models)
- [Model Evaluation](#model-evaluation)
- [Running the App](#running-the-app)
- [Kaggle Authentication](#kaggle-authentication)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

Tuberculosis (TB) remains a significant global health challenge, causing approximately 1.5 million deaths worldwide in 2020. Early and accurate detection is crucial for effective treatment and preventing transmission.

This system leverages state-of-the-art deep learning models to detect tuberculosis from chest X-ray images with high accuracy. It includes:

- **Multiple pre-trained CNN architectures** (ResNet50, VGG16, EfficientNetB0)
- **Comprehensive data preprocessing and augmentation pipeline**
- **Detailed model evaluation with medical-specific metrics**
- **User-friendly Streamlit web interface for predictions**
- **AWS deployment capabilities**

## 🏗️ Project Structure

```
TB_Detection/
├── app/                    # Streamlit application files
│   ├── app.py              # Main Streamlit application
│   ├── ui_components.py    # UI components
│   └── prediction_handler.py # Prediction logic
├── data/                   # Data storage directory
├── models/                 # Model architecture and training
│   ├── model_architectures.py # Model definitions
│   ├── training_pipeline.py  # Training pipeline
│   └── callbacks.py        # Custom training callbacks
├── utils/                  # Utility modules
│   ├── data_loader.py      # Data loading functionality
│   ├── preprocessor.py     # Image preprocessing
│   ├── augmentation.py     # Data augmentation techniques
│   ├── evaluation.py       # Model evaluation utilities
│   ├── metrics.py          # Custom evaluation metrics
│   └── visualization.py    # Visualization tools
├── deployment/             # Deployment resources
│   ├── Dockerfile          # Docker configuration
│   ├── requirements.txt    # Python dependencies
│   └── deploy_aws.py       # AWS deployment script
├── output/                 # Training outputs (models, logs)
├── train.py                # Main training script
├── setup.py                # Setup script for project initialization
└── README.md               # Project documentation
```

## 🧠 Model Architectures

The system incorporates three powerful CNN architectures, each with unique strengths:

### 1. ResNet50
<details>
<summary>Click to expand details</summary>

**Key Features:**
- **Deep Residual Network**: 50 layers with skip connections
- **Residual Blocks**: Address the vanishing gradient problem
- **Excellent Feature Extraction**: Higher-level feature representation
- **ImageNet Pre-trained**: Transfer learning from over 1 million images

**Advantages for TB Detection:**
- Handles complex patterns in X-ray images
- Excellent at identifying subtle abnormalities
- Strong performance on medical imaging tasks
</details>

### 2. VGG16
<details>
<summary>Click to expand details</summary>

**Key Features:**
- **Uniform Architecture**: 16 layers with consistent 3×3 filters
- **Simple Design**: Linear stack of convolutional layers
- **Large Parameter Count**: Rich feature representation
- **Established Performance**: Well-documented on medical images

**Advantages for TB Detection:**
- Consistent feature extraction across image regions
- Good performance with limited training data
- More interpretable feature maps
</details>

### 3. EfficientNetB0
<details>
<summary>Click to expand details</summary>

**Key Features:**
- **Compound Scaling**: Balanced depth, width, and resolution
- **Mobile Inverted Bottleneck**: Efficient architecture
- **Smaller Size**: Fewer parameters than ResNet50 or VGG16
- **State-of-the-art Performance**: High accuracy with lower computation

**Advantages for TB Detection:**
- More efficient training and inference
- Better performance-to-parameter ratio
- Suitable for deployment on resource-constrained devices
</details>

## 🚀 Setup Instructions

### Quick Start with Setup Script

The easiest way to get started is by using the provided setup script, which will:
- Check Python version compatibility
- Install required dependencies
- Create necessary directory structure
- Configure Kaggle API for dataset access (optional)
- Create sample models for demonstration (optional)

```bash
# Activate virtual environment first
python TB_Detection/setup.py
```

Optional arguments:
- `--no-kaggle`: Skip Kaggle API setup
- `--no-sample-models`: Skip creating sample models

### Manual Setup

#### 1. Create and Activate Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Directory Structure Setup

```bash
# Create necessary directories
mkdir -p TB_Detection/data/raw/{train,test}/{tb,normal}
mkdir -p TB_Detection/output/{models,logs,evaluation,visualizations}
```

## 🔧 Training Models

### Training Command

```bash
python train.py --model_name MODEL_NAME [OPTIONS]
```

### Available Models
- `resnet50`: Deep residual network with 50 layers
- `vgg16`: 16-layer CNN with uniform architecture
- `efficientnet`: EfficientNetB0 with compound scaling
- `all`: Train all three models sequentially

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data_dir` | Path to dataset directory | `data` |
| `--output_dir` | Path to save outputs | `output` |
| `--image_size` | Input image size | `224` |
| `--batch_size` | Training batch size | `32` |
| `--epochs` | Number of training epochs | `50` |
| `--learning_rate` | Initial learning rate | `0.001` |
| `--dropout_rate` | Dropout rate | `0.5` |
| `--fine_tune` | Enable model fine-tuning | `False` |
| `--fine_tune_epochs` | Fine-tuning epochs | `30` |
| `--fine_tune_lr` | Fine-tuning learning rate | `0.0001` |
| `--unfreeze_layers` | Layers to unfreeze for fine-tuning | `20` |
| `--use_class_weights` | Use class weights for imbalanced data | `False` |
| `--download_dataset` | Download dataset from Kaggle | `False` |
| `--test_size` | Test set proportion | `0.2` |
| `--valid_size` | Validation set proportion | `0.2` |

### Training Process

1. **Data Preparation**
   - Dataset download (optional)
   - Train/validation/test split
   - Data preprocessing and normalization

2. **Initial Training Phase**
   - Frozen pre-trained base model
   - Addition of custom classification layers
   - Training with specified hyperparameters
   - Early stopping and learning rate reduction

3. **Fine-Tuning Phase** (if enabled)
   - Unfreezing top layers of the base model
   - Training with lower learning rate
   - Further optimization of model weights

4. **Evaluation and Visualization**
   - Performance metrics calculation
   - ROC and precision-recall curves
   - Confusion matrices
   - Class activation maps

### Training Example

```bash
# Basic training with ResNet50
python train.py --model_name resnet50 --epochs 50 --batch_size 32

# Training with fine-tuning and class weights
python train.py --model_name vgg16 --epochs 50 --fine_tune --use_class_weights

# Train all models with extended parameters
python train.py --model_name all --epochs 75 --fine_tune --fine_tune_epochs 40 --batch_size 16 --learning_rate 0.0005
```

## 📊 Model Evaluation

The system evaluates models using comprehensive metrics relevant to medical diagnostics:

- **Accuracy**: Overall correct classification rate
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **Specificity**: True negatives / (True negatives + False positives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **PR AUC**: Area under the Precision-Recall curve
- **Diagnostic Odds Ratio**: Measure of test effectiveness

### Performance Benchmarks

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| ResNet50 | ~94% | ~94% | ~96% | ~94% | ~0.97 |
| VGG16 | ~92% | ~91% | ~95% | ~93% | ~0.96 |
| EfficientNetB0 | ~93% | ~93% | ~94% | ~93% | ~0.97 |

## 🖥️ Running the App

```bash
cd TB_Detection
streamlit run app/app.py
```

The Streamlit app provides:
- X-ray image upload interface
- Model selection options
- Real-time prediction with confidence scores
- Visualizations of results

### ⚠️ Important Note about Models
When first running the app, you'll need either:
1. Trained models at `output/models/{model_name}_final.h5`
2. Sample models created by the setup script

If you see "Model file not found" messages, run:
```bash
python setup.py
```

## ⚙️ Kaggle Authentication

For downloading datasets directly from Kaggle, you need to set up authentication:

1. Create a Kaggle account at https://www.kaggle.com if you don't have one
2. Go to 'Account' page and scroll to 'API' section
3. Click 'Create New API Token' to download kaggle.json
4. Place this file in one of these locations:
   - `~/.kaggle/kaggle.json` (Linux/Mac)
   - `C:\Users\<Windows-username>\.kaggle\kaggle.json` (Windows)

Alternatively, use the setup script which guides you through this process:
```bash
python TB_Detection/setup.py
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Issue: "Model file not found" in the app
**Solution:**
```bash
# Run the setup script to create sample models
python TB_Detection/setup.py

# OR train real models
python TB_Detection/train.py --model_name all
```

# Make sure you're running scripts from the project root
cd /path/to/TB_Detection
python train.py

# Check that your virtual environment is activated
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

#### Issue: "Could not find kaggle.json" when downloading dataset
**Solution:**
```bash
# Run the Kaggle setup script
python TB_Detection/setup.py
```

#### Issue: "Memory error" during training
**Solution:**
```bash
# Try reducing batch size
python TB_Detection/train.py --model_name resnet50 --batch_size 16

# Or reduce image size
python TB_Detection/train.py --model_name resnet50 --image_size 196
``` 