import argparse
import os
import numpy as np
import tensorflow as tf
from utils.preprocessor import ImagePreprocessor
from PIL import Image
import matplotlib.pyplot as plt
import time

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image_path, model_name="resnet50"):
    try:
        preprocessor = ImagePreprocessor(model_name=model_name)
        image = preprocessor.preprocess(image_path)
        return image
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict(model, image):
    try:
        image = np.expand_dims(image, axis=0)
        start_time = time.time()
        prediction = model.predict(image)[0][0]
        end_time = time.time()
        
        return {
            'tb_probability': float(prediction),
            'normal_probability': float(1 - prediction),
            'prediction_time': end_time - start_time
        }
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None

def visualize_prediction(image_path, prediction_result):
    try:
        image = np.array(Image.open(image_path))
        
        tb_prob = prediction_result['tb_probability']
        normal_prob = prediction_result['normal_probability']
        pred_time = prediction_result['prediction_time']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(image)
        ax1.set_title("Input X-ray Image")
        ax1.axis('off')
        
        bars = ['TB', 'Normal']
        heights = [tb_prob, normal_prob]
        colors = ['red', 'green']
        
        ax2.bar(bars, heights, color=colors)
        ax2.set_ylim(0, 1)
        result = "TB Detected" if tb_prob >= 0.5 else "Normal"
        ax2.set_title(f"Prediction: {result}")
        ax2.set_ylabel("Probability")
        
        for i, v in enumerate(heights):
            ax2.text(i, v + 0.02, f'{v:.2%}', ha='center')
            
        plt.figtext(0.5, 0.01, f"Prediction time: {pred_time:.3f} seconds", 
                   ha="center", fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error visualizing prediction: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Test TB Detection Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the X-ray image for testing")
    parser.add_argument("--model_name", type=str, default="resnet50", 
                        choices=["resnet50", "vgg16", "efficientnet"],
                        help="Model architecture")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Model path not found: {args.model_path}")
        return
        
    if not os.path.exists(args.image_path):
        print(f"Image path not found: {args.image_path}")
        return
        
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    if model is None:
        return
        
    print(f"Preprocessing image {args.image_path}...")
    image = preprocess_image(args.image_path, args.model_name)
    
    if image is None:
        return
        
    print("Making prediction...")
    prediction_result = predict(model, image)
    
    if prediction_result is None:
        return
        
    tb_prob = prediction_result['tb_probability']
    normal_prob = prediction_result['normal_probability']
    pred_time = prediction_result['prediction_time']
    
    print("\n" + "=" * 40)
    print("PREDICTION RESULTS")
    print("=" * 40)
    print(f"TB Probability:     {tb_prob:.2%}")
    print(f"Normal Probability: {normal_prob:.2%}")
    print(f"Prediction Time:    {pred_time:.3f} seconds")
    print(f"Result:             {'TB Detected' if tb_prob >= 0.5 else 'Normal'}")
    print("=" * 40)
    
    visualize_prediction(args.image_path, prediction_result)
    
if __name__ == "__main__":
    main() 