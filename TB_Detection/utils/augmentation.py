import tensorflow as tf
import numpy as np
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataAugmenter:
    def __init__(self, rotation_range=15, 
                horizontal_flip=True,
                zoom_range=(0.8, 1.2),
                width_shift_range=0.1,
                height_shift_range=0.1,
                brightness_range=(0.8, 1.2),
                fill_mode='nearest'):
        
        self.keras_augmenter = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            brightness_range=brightness_range,
            fill_mode=fill_mode
        )
        
        self.albumentation_transform = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),            
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
        
        self.simple_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])
    
    def apply_keras_augmentation(self, image):
        image = image.reshape((1,) + image.shape)
        augmented = self.keras_augmenter.flow(image, batch_size=1)[0]
        return augmented
    
    def apply_albumentation(self, image, aggressive=False):
        if aggressive:
            augmented = self.albumentation_transform(image=image)['image']
        else:
            augmented = self.simple_transform(image=image)['image']
        return augmented
    
    def create_tf_augmentation_layer(self):
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])
    
    def augment_tf_dataset(self, dataset, augment_class_0=True, augment_class_1=True):
        def augment_map_fn(x, y):
            if (augment_class_0 and y == 0) or (augment_class_1 and y == 1):
                x = tf.py_function(
                    lambda img: self.apply_keras_augmentation(img.numpy()),
                    [x],
                    tf.float32
                )
            return x, y
            
        return dataset.map(augment_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    def generate_augmented_batch(self, images, labels, batch_size=32, class_specific=None):
        augmented_images = []
        augmented_labels = []
        
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            
            if class_specific is not None and label != class_specific:
                continue
                
            augmented_images.append(image)
            augmented_labels.append(label)
            
            num_augmentations = 3 if class_specific is not None else 1
            
            for _ in range(num_augmentations):
                aug_image = self.apply_albumentation(image, aggressive=(class_specific is not None))
                augmented_images.append(aug_image)
                augmented_labels.append(label)
                
            if len(augmented_images) >= batch_size:
                break
                
        return np.array(augmented_images[:batch_size]), np.array(augmented_labels[:batch_size]) 