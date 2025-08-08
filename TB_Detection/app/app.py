

import os
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(project_root)

try:
    from utils.preprocessor import ImagePreprocessor
    print("Successfully imported using direct path")
except ImportError:
    try:
        from TB_Detection.utils.preprocessor import ImagePreprocessor
        print("Successfully imported using TB_Detection prefix")
    except ImportError:
        sys.path.append(os.path.join(parent_dir, 'utils'))
        from preprocessor import ImagePreprocessor
        print("Successfully imported using utils path")

class TBDetectionApp:
    def __init__(self, model_paths=None, model_names=None):
        self.setup_directories()
        
        local_model_dir = os.path.join(parent_dir, "output", "models")
        
        self.model_paths = model_paths or {
            "ResNet50": os.path.join(local_model_dir, "resnet50_final.h5"),
            "VGG16": os.path.join(local_model_dir, "vgg16_final.h5"),
            "EfficientNetB0": os.path.join(local_model_dir, "efficientnet_final.h5")
        }
        self.model_names = model_names or list(self.model_paths.keys())
        self.models = {}
        self.preprocessors = {}
        self.current_model_name = None
    
    def setup_directories(self):
        dirs_to_create = [
            os.path.join(parent_dir, "output"),
            os.path.join(parent_dir, "output", "models")
        ]
        
        for directory in dirs_to_create:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
                
    def load_models(self):
        models_found = False
        
        for name, path in self.model_paths.items():
            if os.path.exists(path):
                models_found = True
                try:
                    self.models[name] = tf.keras.models.load_model(path)
                    if name == "ResNet50":
                        self.preprocessors[name] = ImagePreprocessor(model_name="resnet50")
                    elif name == "VGG16":
                        self.preprocessors[name] = ImagePreprocessor(model_name="vgg16")
                    elif name == "EfficientNetB0":
                        self.preprocessors[name] = ImagePreprocessor(model_name="efficientnet")
                    print(f"Successfully loaded model: {name} from {path}")
                except Exception as e:
                    st.error(f"Error loading model {name}: {str(e)}")
            else:
                st.warning(f"Model file not found for {name}: {path}")
        
        return models_found
                
    def get_available_models(self):
        return [name for name in self.model_names if name in self.models]
                
    def preprocess_image(self, image, model_name):
        preprocessor = self.preprocessors.get(model_name)
        if preprocessor is None:
            st.error(f"No preprocessor available for model {model_name}")
            return None
            
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            st.error("Unsupported image format")
            return None
            
        return preprocessor.preprocess(img)
        
    def predict(self, image, model_name):
        if model_name not in self.models:
            st.error(f"Model {model_name} not loaded")
            return None
            
        preprocessed = self.preprocess_image(image, model_name)
        
        if preprocessed is None:
            return None
            
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        try:
            prediction = self.models[model_name].predict(preprocessed)[0][0]
            return float(prediction)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="TB Detection System",
        page_icon="🫁",
        layout="wide"
    )
    
    app = TBDetectionApp()
    
    st.title("Tuberculosis Detection System")
    st.markdown("""
    This application uses deep learning to detect tuberculosis from chest X-ray images.
    Upload a chest X-ray image to get a prediction.
    """)
    
    with st.spinner("Loading models..."):
        models_found = app.load_models()
        
    available_models = app.get_available_models()
    
    if not available_models:
        st.error("No models are available. Please check model paths and files.")
        
        st.subheader("How to Train Models")
        st.info("You need to train models first using the train.py script before using the app.")
        
        st.code("""
# Train ResNet50 model
python train.py --model_name resnet50 --epochs 50 --batch_size 32 --use_class_weights --fine_tune

# Train VGG16 model
python train.py --model_name vgg16 --epochs 50 --batch_size 32 --use_class_weights --fine_tune

# Train EfficientNetB0 model
python train.py --model_name efficientnet --epochs 50 --batch_size 32 --use_class_weights --fine_tune
        """)

        st.subheader("Expected Model Paths")
        for name, path in app.model_paths.items():
            st.code(f"{name}: {path}")
            
        return
        
    with st.sidebar:
        st.header("Settings")
        
        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=0 if available_models else None,
            help="Choose the deep learning model for prediction"
        )
        
        if selected_model:
            app.current_model_name = selected_model
        
        st.divider()
        st.subheader("About")
        st.markdown("""
        This application detects tuberculosis from chest X-ray images using 
        deep learning models trained on thousands of X-ray images.
        
        **Models available:**
        - ResNet50: Deep residual network
        - VGG16: Classic CNN architecture
        - EfficientNetB0: Efficient and accurate network
        
        **Disclaimer:** This tool is for educational purposes only and should not be used for medical diagnosis.
        """)
        
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload X-ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a chest X-ray image for TB detection"
        )
        
        if uploaded_file is not None and available_models:
            try:
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array, img_array, img_array], axis=-1)
                elif img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                    
                st.image(img_array, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Predict"):
                    with st.spinner("Processing image..."):
                        start_time = time.time()
                        prediction = app.predict(img_array, app.current_model_name)
                        end_time = time.time()
                        
                        if prediction is not None:
                            with col2:
                                st.subheader("Detection Results")
                                
                                tb_probability = prediction
                                normal_probability = 1 - prediction
                                
                                result_text = "TB Detected" if tb_probability >= 0.5 else "Normal"
                                result_color = "red" if tb_probability >= 0.5 else "green"
                                
                                st.markdown(f"<h2 style='color: {result_color};'>{result_text}</h2>", unsafe_allow_html=True)
                                
                                st.subheader("Probability Scores:")
                                st.progress(tb_probability)
                                st.write(f"TB: {tb_probability:.2%}")
                                
                                st.progress(normal_probability)
                                st.write(f"Normal: {normal_probability:.2%}")
                                
                                st.info(f"Processed in {end_time - start_time:.2f} seconds using {app.current_model_name}")
                                
                                st.markdown("""
                                **Note:** This is an AI-assisted prediction and should not be used for diagnosis. 
                                Always consult with a qualified medical professional.
                                """)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 