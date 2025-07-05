# Tuberculosis Detection System

This system uses deep learning models to detect tuberculosis from chest X-ray images.

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Kaggle Authentication Setup

To download the dataset automatically, you need to set up Kaggle authentication:

1. Create a Kaggle account at https://www.kaggle.com if you don't have one
2. Go to Account -> API -> Create New API Token to download `kaggle.json`
3. Place the `kaggle.json` file in:
   - Windows: `C:\Users\YOUR_USERNAME\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`
4. Ensure the file has appropriate permissions (600 on Linux/Mac)

Alternatively, you can manually download the dataset from Kaggle:
1. Visit https://www.kaggle.com/datasets/yasserhessein/tuberculosis-chest-x-rays-images
2. Download and extract the dataset to the `TB_Detection/data` directory

### 4. Run the Application

```bash
cd TB_Detection
streamlit run app/app.py
```

## Project Structure

- `app/`: Streamlit web application for TB detection
- `utils/`: Utility modules for data processing and model handling
- `models/`: Directory for trained models
- `data/`: Dataset directory
- `output/`: Training outputs and saved models

## Models

The system uses three pre-trained models:
- ResNet50
- VGG16
- EfficientNetB0

## Training

To train the models:

```bash
python train.py --model_name resnet50 --epochs 50 --batch_size 32 --use_class_weights --fine_tune
```

## About the Dataset

The dataset contains chest X-ray images categorized into two classes:
- Normal: Healthy chest X-rays
- TB: X-rays showing tuberculosis manifestations

## Troubleshooting

### Kaggle API Error
If you see `Could not find kaggle.json` error:
- Ensure you've created and placed the kaggle.json file in the correct location
- Check file permissions (600 on Linux/Mac)
- Consider manually downloading the dataset as described above

### Import Errors
If you encounter module import errors:
- Ensure you're running the app from the correct directory
- Verify the project structure is maintained
- Make sure all dependencies are installed 