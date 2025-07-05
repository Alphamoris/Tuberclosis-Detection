import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import os
import cv2
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import pandas as pd
import time

class Visualizer:
    def __init__(self, output_dir='./visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_training_history(self, history, metrics=['loss', 'accuracy', 'precision', 'recall', 'auc']):
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics*5, 5))
        
        for i, metric in enumerate(metrics):
            ax = axes[i] if n_metrics > 1 else axes
            ax.plot(history.history[metric], label=f'Train {metric}')
            
            if f'val_{metric}' in history.history:
                ax.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
                
            ax.set_title(f'Model {metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.legend()
            
        plt.tight_layout()
        timestamp = int(time.time())
        plt.savefig(os.path.join(self.output_dir, f'training_history_{timestamp}.png'), dpi=300)
        plt.close()
        
    def plot_sample_predictions(self, model, test_dataset, class_names=['Normal', 'TB'], num_samples=10):
        all_images = []
        all_labels = []
        all_predictions = []
        
        for images, labels in test_dataset:
            predictions = model.predict(images)
            
            all_images.append(images.numpy())
            all_labels.append(labels.numpy())
            all_predictions.append(predictions)
            
            if len(all_images) * images.shape[0] >= num_samples:
                break
                
        all_images = np.vstack([img for img in all_images])[:num_samples]
        all_labels = np.vstack([label for label in all_labels])[:num_samples].flatten()
        all_predictions = np.vstack([pred for pred in all_predictions])[:num_samples].flatten()
        
        plt.figure(figsize=(20, num_samples * 2))
        
        for i in range(num_samples):
            plt.subplot(num_samples // 5 + 1, 5, i + 1)
            
            image = all_images[i]
            
            if image.shape[-1] == 3:
                plt.imshow(image)
            else:
                plt.imshow(image[:,:,0], cmap='gray')
                
            true_class = class_names[int(all_labels[i])]
            pred_class = class_names[int(all_predictions[i] > 0.5)]
            confidence = all_predictions[i] if pred_class == class_names[1] else 1 - all_predictions[i]
            
            color = 'green' if true_class == pred_class else 'red'
            plt.title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}", color=color)
            plt.axis('off')
            
        plt.tight_layout()
        timestamp = int(time.time())
        plt.savefig(os.path.join(self.output_dir, f'sample_predictions_{timestamp}.png'), dpi=300)
        plt.close()
        
    def plot_confusion_matrix(self, y_true, y_pred, class_names=['Normal', 'TB']):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        timestamp = int(time.time())
        plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{timestamp}.png'), dpi=300)
        plt.close()
        
        return cm
        
    def plot_roc_curve(self, y_true, y_pred_probs):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        
        timestamp = int(time.time())
        plt.savefig(os.path.join(self.output_dir, f'roc_curve_{timestamp}.png'), dpi=300)
        plt.close()
        
        return roc_auc
        
    def plot_precision_recall_curve(self, y_true, y_pred_probs):
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        timestamp = int(time.time())
        plt.savefig(os.path.join(self.output_dir, f'pr_curve_{timestamp}.png'), dpi=300)
        plt.close()
        
        return pr_auc
        
    def plot_gradcam(self, model, img, layer_name=None, class_idx=None):
        if layer_name is None:
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4 and 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
                    
        if layer_name is None:
            raise ValueError("Could not find convolutional layer for GradCAM")
            
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(np.array([img]))
            if class_idx is None:
                class_idx = np.argmax(predictions[0])
            loss = predictions[:, class_idx]
            
        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_output = conv_output[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        img_rgb = np.copy(img)
        
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = cv2.addWeighted(cv2.cvtColor(np.uint8(img_rgb*255), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap)
        axes[1].set_title('Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title('GradCAM Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        timestamp = int(time.time())
        plt.savefig(os.path.join(self.output_dir, f'gradcam_{timestamp}.png'), dpi=300)
        plt.close()
        
        return superimposed_img
        
    def plot_model_comparison(self, metrics_dict, metrics=['accuracy', 'precision', 'recall', 'f1_score', 'specificity']):
        df = pd.DataFrame(metrics_dict).T
        
        plt.figure(figsize=(12, 6))
        df[metrics].plot(kind='bar', figsize=(12, 6))
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.ylim([0, 1])
        plt.xlabel('Model')
        plt.grid(axis='y')
        plt.legend(loc='lower right')
        
        timestamp = int(time.time())
        plt.savefig(os.path.join(self.output_dir, f'model_comparison_{timestamp}.png'), dpi=300)
        plt.close()
        
        return df
        
    def plot_class_distribution(self, dataset_splits):
        labels = []
        datasets = []
        
        for split_name, split_data in dataset_splits.items():
            for class_name, count in split_data.items():
                labels.append(class_name)
                datasets.append(split_name)
                
        df = pd.DataFrame({'Dataset': datasets, 'Class': labels, 'Count': list(split_data.values())})
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Class', y='Count', hue='Dataset', data=df)
        plt.title('Class Distribution Across Datasets')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        timestamp = int(time.time())
        plt.savefig(os.path.join(self.output_dir, f'class_distribution_{timestamp}.png'), dpi=300)
        plt.close()
        
        return df 