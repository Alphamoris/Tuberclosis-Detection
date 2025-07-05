import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import os

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), model_name='resnet50'):
        self.target_size = target_size
        self.model_name = model_name.lower()
        
    def _apply_model_specific_preprocessing(self, img):
        if self.model_name == 'resnet50':
            return resnet_preprocess(img)
        elif self.model_name == 'vgg16':
            return vgg_preprocess(img)
        elif self.model_name == 'efficientnet':
            return efficientnet_preprocess(img)
        else:
            return img / 255.0
            
    def preprocess(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img = cv2.resize(img, self.target_size)
        img = np.array(img, dtype=np.float32)
        img = self._apply_model_specific_preprocessing(img)
        
        return img
        
    def preprocess_dataset(self, dataset):
        def preprocess_batch(images, labels):
            preprocessed_images = tf.py_function(
                lambda x: np.array([self.preprocess(img) for img in x.numpy()]),
                [images],
                tf.float32
            )
            preprocessed_images.set_shape([None, self.target_size[0], self.target_size[1], 3])
            return preprocessed_images, labels
            
        return dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
        
    def quality_check(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Image cannot be read"
                
            height, width, channels = img.shape
            if height < 10 or width < 10:
                return False, "Image dimensions too small"
                
            if channels != 3:
                return False, "Image does not have 3 channels"
                
            return True, "Image passed quality check"
        except Exception as e:
            return False, f"Error during quality check: {str(e)}"
            
    def extract_metadata(self, image_path):
        try:
            img = cv2.imread(image_path)
            height, width, channels = img.shape
            file_size = os.path.getsize(image_path)
            mean_pixel = np.mean(img)
            std_pixel = np.std(img)
            
            return {
                "height": height,
                "width": width,
                "channels": channels,
                "file_size_kb": file_size / 1024,
                "mean_pixel_value": mean_pixel,
                "std_pixel_value": std_pixel
            }
        except Exception as e:
            return {"error": str(e)}
            
    def visualize_preprocessing(self, image_path):
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        preprocessed = self.preprocess(image_path)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        
        if self.model_name != 'resnet50':
            plt.imshow(preprocessed)
        else:
            temp = preprocessed.copy()
            temp = temp - np.min(temp)
            temp = temp / np.max(temp)
            plt.imshow(temp)
            
        plt.title("Preprocessed Image")
        plt.show() 