import os
import zipfile
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import glob

class DataLoader:
    def __init__(self, dataset_path=None, dataset_name='yasserhessein/tuberculosis-chest-x-rays-images'):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path if dataset_path else 'data/'
        self.train_dir = os.path.join(self.dataset_path, 'train')
        self.valid_dir = os.path.join(self.dataset_path, 'valid')
        self.test_dir = os.path.join(self.dataset_path, 'test')
        
    def download_dataset(self):
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(self.dataset_name, path=self.dataset_path, unzip=True)
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("If you're seeing Kaggle authentication errors, please follow these steps:")
            print("1. Create a Kaggle account at https://www.kaggle.com")
            print("2. Go to Account -> API -> Create New API Token to download kaggle.json")
            print("3. Place kaggle.json in ~/.kaggle/ directory (C:\\Users\\username\\.kaggle\\ on Windows)")
            print("4. Ensure the file has appropriate permissions (600 on Linux/Mac)")
            return False
    
    def extract_dataset(self, zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_path)
    
    def organize_data(self, tb_dir='TB', normal_dir='Normal', test_size=0.2, valid_size=0.2, random_state=42):
        print(f"Looking for TB images in: {tb_dir}")
        print(f"Looking for Normal images in: {normal_dir}")
        
        tb_images = glob.glob(os.path.join(tb_dir, '**', '*.jpg'), recursive=True)
        tb_images.extend(glob.glob(os.path.join(tb_dir, '**', '*.jpeg'), recursive=True))
        tb_images.extend(glob.glob(os.path.join(tb_dir, '**', '*.png'), recursive=True))
        
        normal_images = glob.glob(os.path.join(normal_dir, '**', '*.jpg'), recursive=True)
        normal_images.extend(glob.glob(os.path.join(normal_dir, '**', '*.jpeg'), recursive=True))
        normal_images.extend(glob.glob(os.path.join(normal_dir, '**', '*.png'), recursive=True))
        
        print(f"Found {len(tb_images)} TB images and {len(normal_images)} Normal images")
        
        if len(tb_images) == 0 or len(normal_images) == 0:
            print("ERROR: No images found in one of the directories!")
            return None
        
        test_size = min(test_size, 0.3) 
        valid_size = min(valid_size, 0.3) 
        
        tb_train_valid, tb_test = train_test_split(tb_images, test_size=test_size, random_state=random_state)
        tb_train, tb_valid = train_test_split(tb_train_valid, test_size=valid_size/(1-test_size), random_state=random_state)
        
        normal_train_valid, normal_test = train_test_split(normal_images, test_size=test_size, random_state=random_state)
        normal_train, normal_valid = train_test_split(normal_train_valid, test_size=valid_size/(1-test_size), random_state=random_state)
        
        print(f"Split sizes - TB: {len(tb_train)} train, {len(tb_valid)} valid, {len(tb_test)} test")
        print(f"Split sizes - Normal: {len(normal_train)} train, {len(normal_valid)} valid, {len(normal_test)} test")
        
        os.makedirs(os.path.join(self.train_dir, 'TB'), exist_ok=True)
        os.makedirs(os.path.join(self.train_dir, 'Normal'), exist_ok=True)
        os.makedirs(os.path.join(self.valid_dir, 'TB'), exist_ok=True)
        os.makedirs(os.path.join(self.valid_dir, 'Normal'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'TB'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'Normal'), exist_ok=True)
        
        for split, tb_imgs, normal_imgs in [
            ('train', tb_train, normal_train),
            ('valid', tb_valid, normal_valid),
            ('test', tb_test, normal_test)
        ]:
            for img_path in tb_imgs:
                dest = os.path.join(self.dataset_path, split, 'TB', os.path.basename(img_path))
                try:
                    tf.io.gfile.copy(img_path, dest, overwrite=True)
                except Exception as e:
                    print(f"Error copying {img_path}: {e}")
                
            for img_path in normal_imgs:
                dest = os.path.join(self.dataset_path, split, 'Normal', os.path.basename(img_path))
                try:
                    tf.io.gfile.copy(img_path, dest, overwrite=True)
                except Exception as e:
                    print(f"Error copying {img_path}: {e}")
                
        return {
            'train': {'TB': len(tb_train), 'Normal': len(normal_train)},
            'valid': {'TB': len(tb_valid), 'Normal': len(normal_valid)},
            'test': {'TB': len(tb_test), 'Normal': len(normal_test)}
        }
    
    def get_class_weights(self):
        train_tb_count = len(os.listdir(os.path.join(self.train_dir, 'TB')))
        train_normal_count = len(os.listdir(os.path.join(self.train_dir, 'Normal')))
        
        total = train_tb_count + train_normal_count
        weight_for_0 = (1 / train_normal_count) * (total / 2.0)
        weight_for_1 = (1 / train_tb_count) * (total / 2.0)
        
        return {0: weight_for_0, 1: weight_for_1}
    
    def create_tf_datasets(self, img_height=224, img_width=224, batch_size=32):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_dir,
            validation_split=None,
            seed=42,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='binary'
        )
        
        valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.valid_dir,
            validation_split=None,
            seed=42,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='binary'
        )
        
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.test_dir,
            validation_split=None,
            seed=42,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='binary'
        )
        
        return train_ds, valid_ds, test_ds 