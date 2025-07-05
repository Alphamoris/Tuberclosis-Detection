import os
import sys
import argparse
import subprocess
import tensorflow as tf
import numpy as np
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    required_packages = [
        "tensorflow", "numpy", "pandas", "matplotlib", "pillow",
        "opencv-python", "scikit-learn", "streamlit"
    ]
    
    kaggle_available = False
    try:
        import kaggle
        print(f"✓ kaggle")
        kaggle_available = True
    except (ImportError, OSError):
        print(f"✗ kaggle (Will be skipped)")
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print("\nMissing packages detected. Installing...")
        subprocess.call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("Dependencies installed.")
    else:
        print("\nCore dependencies are installed.")
        
    return kaggle_available

def create_directories():
    """Create necessary directories"""
    print("\nCreating directory structure...")
    dirs = [
        "data/raw/train/tb",
        "data/raw/train/normal",
        "data/raw/test/tb",
        "data/raw/test/normal",
        "output/models",
        "output/logs",
        "output/evaluation",
        "output/visualizations"
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(current_dir, d), exist_ok=True)
        print(f"✓ {d}")
    
    print("Directory structure created.")

def create_sample_model():
    """Create a small sample model that can be saved to the correct location"""
    print("\nCreating sample models...")
    
    models_to_create = {
        "resnet50": "ResNet50",
        "vgg16": "VGG16",
        "efficientnet": "EfficientNetB0"
    }
    
    os.makedirs(os.path.join(current_dir, "output", "models"), exist_ok=True)
    
    for model_key, model_name in models_to_create.items():
        print(f"Creating {model_name} sample model...")
        
        try:
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            model = tf.keras.Model(inputs, outputs)
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            output_path = os.path.join(current_dir, "output", "models", f"{model_key}_final.h5")
            model.save(output_path)
            print(f"✓ Saved sample {model_name} model to {output_path}")
        except Exception as e:
            print(f"✗ Error creating {model_name} model: {str(e)}")
    
    print("Sample models creation completed.")

def setup_kaggle_api():
    """Setup Kaggle API for dataset download"""
    print("\nSetting up Kaggle API access...")
    
    kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    kaggle_api_path = os.path.join(kaggle_dir, 'kaggle.json')
    
    if os.path.exists(kaggle_api_path):
        print("✓ Kaggle API credentials already exist.")
        return True
        
    print("To download datasets from Kaggle, you need to provide your API credentials.")
    print("Please visit https://www.kaggle.com/settings/account to generate an API token.")
    print("Then, paste your API credentials in the format: {\"username\":\"your-username\",\"key\":\"your-key\"}")
    
    try:
        username = input("Enter your Kaggle username: ")
        key = input("Enter your Kaggle API key: ")
        
        if not username or not key:
            print("Username or key is empty. Skipping Kaggle setup.")
            return False
            
        import json
        with open(kaggle_api_path, 'w') as f:
            json.dump({"username": username, "key": key}, f)
            
        os.chmod(kaggle_api_path, 0o600)
        print("✓ Kaggle API credentials configured.")
        return True
        
    except Exception as e:
        print(f"Error setting up Kaggle API: {str(e)}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    import platform
    print(f"Python version: {platform.python_version()}")
    major, minor = map(int, platform.python_version_tuple()[:2])
    
    if major < 3 or (major == 3 and minor < 8):
        print("Warning: This project is recommended to run on Python 3.8 or newer.")
    else:
        print("✓ Python version is compatible.")

def main():
    parser = argparse.ArgumentParser(description="Setup the TB Detection project")
    parser.add_argument('--no-kaggle', action='store_true', help="Skip Kaggle API setup")
    parser.add_argument('--no-sample-models', action='store_true', help="Skip creating sample models")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("TB Detection System Setup")
    print("="*60)
    
    check_python_version()
    
    kaggle_available = check_dependencies()
    
    create_directories()
    
    if not args.no_kaggle and kaggle_available:
        setup_kaggle_api()
                    
    if not args.no_sample_models:
        try:
            create_sample_model()
        except ImportError as e:
            print(f"\nError creating sample models: {str(e)}")
            print("You may need to install additional dependencies.")
        except Exception as e:
            print(f"\nError creating sample models: {str(e)}")
    
    print("\n" + "="*60)
    print("Setup completed successfully!")
    print("""
To run the TB Detection app:
    streamlit run TB_Detection/app/app.py
    
To train models with real data:
    python TB_Detection/train.py --model_name resnet50 --epochs 50 --batch_size 32
    """)
    print("="*60)

if __name__ == "__main__":
    main() 