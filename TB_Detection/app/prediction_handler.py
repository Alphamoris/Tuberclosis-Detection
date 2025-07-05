import numpy as np
import tensorflow as tf
import cv2
import os
import time
import json
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import io
import base64

sys.path.append("..")
from utils.preprocessor import ImagePreprocessor
import utils.metrics as metrics_utils

class PredictionHandler:
    def __init__(self, models_dict=None, model_paths=None):
        self.models = models_dict or {}
        self.model_paths = model_paths or {}
        self.preprocessors = {}
        self.results_history = []
        self.load_counter = 0
        
    def load_model(self, model_name, model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            self.models[model_name] = model
            
            if "resnet" in model_name.lower():
                self.preprocessors[model_name] = ImagePreprocessor(model_name="resnet50")
            elif "vgg" in model_name.lower():
                self.preprocessors[model_name] = ImagePreprocessor(model_name="vgg16")
            elif "efficient" in model_name.lower():
                self.preprocessors[model_name] = ImagePreprocessor(model_name="efficientnet")
            else:
                self.preprocessors[model_name] = ImagePreprocessor()
                
            self.load_counter += 1
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return False
            
    def load_all_models(self):
        results = {}
        for name, path in self.model_paths.items():
            results[name] = self.load_model(name, path)
        return results
        
    def preprocess_image(self, image, model_name):
        if model_name not in self.preprocessors:
            print(f"No preprocessor found for model {model_name}")
            return None
            
        preprocessor = self.preprocessors[model_name]
        
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
        else:
            print("Unsupported image format")
            return None
            
        return preprocessor.preprocess(img)
        
    def predict_with_model(self, image, model_name):
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
            
        preprocessed = self.preprocess_image(image, model_name)
        
        if preprocessed is None:
            return None
            
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        try:
            start_time = time.time()
            prediction = self.models[model_name].predict(preprocessed)[0][0]
            end_time = time.time()
            
            result = {
                'model_name': model_name,
                'tb_probability': float(prediction),
                'normal_probability': float(1 - prediction),
                'prediction_time': end_time - start_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            print(f"Error during prediction with model {model_name}: {str(e)}")
            return None
            
    def predict_with_all_models(self, image):
        results = {}
        for model_name in self.models:
            result = self.predict_with_model(image, model_name)
            if result:
                results[model_name] = result
                
        return results
        
    def generate_report(self, prediction_result):
        if not prediction_result:
            return "Error: No prediction result available."
            
        model_name = prediction_result.get('model_name', 'Unknown')
        tb_prob = prediction_result.get('tb_probability', 0)
        normal_prob = prediction_result.get('normal_probability', 0)
        pred_time = prediction_result.get('prediction_time', 0)
        timestamp = prediction_result.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        prediction = "TB Positive" if tb_prob >= 0.5 else "Normal"
        confidence = tb_prob if tb_prob >= 0.5 else normal_prob
        
        report = [
            f"TB Detection Report - {timestamp}",
            f"=" * 40,
            f"Model: {model_name}",
            f"Prediction: {prediction}",
            f"Confidence: {confidence:.2%}",
            f"",
            f"Probabilities:",
            f"  TB: {tb_prob:.2%}",
            f"  Normal: {normal_prob:.2%}",
            f"",
            f"Processing Time: {pred_time:.3f} seconds",
            f"",
            f"Note: This is an AI-assisted prediction and should not be",
            f"used for clinical diagnosis. Always consult with a qualified",
            f"medical professional for proper diagnosis and treatment."
        ]
        
        return "\n".join(report)
        
    def save_result(self, image, prediction_result, output_dir="./results"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        img_path = os.path.join(output_dir, f"image_{timestamp}.png")
        result_path = os.path.join(output_dir, f"result_{timestamp}.json")
        report_path = os.path.join(output_dir, f"report_{timestamp}.txt")
        visualization_path = os.path.join(output_dir, f"visual_{timestamp}.png")
        
        if isinstance(image, np.ndarray):
            cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
        with open(result_path, 'w') as f:
            json.dump(prediction_result, f, indent=4)
            
        report = self.generate_report(prediction_result)
        with open(report_path, 'w') as f:
            f.write(report)
            
        self.create_visualization(image, prediction_result, visualization_path)
        
        return {
            'image_path': img_path,
            'result_path': result_path,
            'report_path': report_path,
            'visualization_path': visualization_path
        }
        
    def create_visualization(self, image, prediction_result, output_path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        tb_prob = prediction_result.get('tb_probability', 0)
        normal_prob = prediction_result.get('normal_probability', 0)
        prediction = "TB Positive" if tb_prob >= 0.5 else "Normal"
        
        ax1.imshow(image)
        ax1.set_title("Original X-ray")
        ax1.axis('off')
        
        bars = ['TB', 'Normal']
        heights = [tb_prob, normal_prob]
        colors = ['red', 'green']
        
        ax2.bar(bars, heights, color=colors)
        ax2.set_ylim(0, 1)
        ax2.set_title(f"Prediction: {prediction}")
        ax2.set_ylabel("Probability")
        
        for i, v in enumerate(heights):
            ax2.text(i, v + 0.02, f'{v:.2%}', ha='center')
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
    def get_history(self, limit=10):
        return self.results_history[-limit:]
        
    def get_visualizer(self, image, prediction_result):
        tb_prob = prediction_result.get('tb_probability', 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image)
        
        result = "TB Positive" if tb_prob >= 0.5 else "Normal"
        color = "red" if tb_prob >= 0.5 else "green"
        
        plt.title(f"{result} ({tb_prob:.1%} confidence)", color=color, fontsize=16)
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        return buf 