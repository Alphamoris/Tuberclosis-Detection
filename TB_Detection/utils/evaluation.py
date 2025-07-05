import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd
import os
import time

class ModelEvaluator:
    def __init__(self, model, test_dataset, output_dir='./output/evaluation'):
        self.model = model
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def predict_all(self):
        all_images = []
        all_labels = []
        all_predictions = []
        
        for images, labels in self.test_dataset:
            predictions = self.model.predict(images)
            
            all_images.extend(images.numpy())
            all_labels.extend(labels.numpy().flatten())
            all_predictions.extend(predictions.flatten())
            
        return np.array(all_images), np.array(all_labels), np.array(all_predictions)
        
    def calculate_metrics(self, y_true, y_pred_probs, threshold=0.5):
        y_pred = (y_pred_probs >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(recall_curve, precision_curve)
        
        ppv = precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        diagnostics_odds_ratio = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'diagnostics_odds_ratio': diagnostics_odds_ratio,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
    def plot_confusion_matrix(self, y_true, y_pred_probs, threshold=0.5):
        y_pred = (y_pred_probs >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'TB'], 
                   yticklabels=['Normal', 'TB'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        timestamp = int(time.time())
        save_path = os.path.join(self.output_dir, f'confusion_matrix_{threshold}_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()
        
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
        save_path = os.path.join(self.output_dir, f'roc_curve_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()
        
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
        save_path = os.path.join(self.output_dir, f'pr_curve_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()
        
    def plot_threshold_analysis(self, y_true, y_pred_probs):
        thresholds = np.arange(0.1, 1.0, 0.1)
        
        metrics = []
        for threshold in thresholds:
            y_pred = (y_pred_probs >= threshold).astype(int)
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp)
            
            metrics.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        plt.figure(figsize=(12, 8))
        plt.plot(metrics_df['threshold'], metrics_df['accuracy'], label='Accuracy')
        plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision')
        plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall')
        plt.plot(metrics_df['threshold'], metrics_df['f1_score'], label='F1-Score')
        plt.plot(metrics_df['threshold'], metrics_df['specificity'], label='Specificity')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs. Threshold')
        plt.legend()
        plt.grid(True)
        
        timestamp = int(time.time())
        save_path = os.path.join(self.output_dir, f'threshold_analysis_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()
        
        return metrics_df
        
    def find_optimal_threshold(self, y_true, y_pred_probs, metric='f1_score'):
        thresholds = np.arange(0.01, 1.0, 0.01)
        
        scores = []
        for threshold in thresholds:
            y_pred = (y_pred_probs >= threshold).astype(int)
            
            if metric == 'f1_score':
                score = f1_score(y_true, y_pred)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            scores.append((threshold, score))
            
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        optimal_threshold, best_score = sorted_scores[0]
        
        return optimal_threshold, best_score
        
    def evaluate(self):
        _, y_true, y_pred_probs = self.predict_all()
        
        default_metrics = self.calculate_metrics(y_true, y_pred_probs)
        
        self.plot_confusion_matrix(y_true, y_pred_probs)
        self.plot_roc_curve(y_true, y_pred_probs)
        self.plot_precision_recall_curve(y_true, y_pred_probs)
        threshold_df = self.plot_threshold_analysis(y_true, y_pred_probs)
        
        optimal_threshold, best_f1 = self.find_optimal_threshold(y_true, y_pred_probs, 'f1_score')
        optimal_metrics = self.calculate_metrics(y_true, y_pred_probs, optimal_threshold)
        
        self.plot_confusion_matrix(y_true, y_pred_probs, optimal_threshold)
        
        results = {
            'default_metrics': default_metrics,
            'optimal_threshold': optimal_threshold,
            'optimal_metrics': optimal_metrics,
            'threshold_analysis': threshold_df.to_dict('records')
        }
        
        timestamp = int(time.time())
        with open(os.path.join(self.output_dir, f'evaluation_report_{timestamp}.txt'), 'w') as f:
            f.write("TB DETECTION MODEL EVALUATION\n")
            f.write("============================\n\n")
            
            f.write("DEFAULT THRESHOLD (0.5):\n")
            for metric, value in default_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nOPTIMAL THRESHOLD:\n")
            f.write(f"Threshold: {optimal_threshold:.4f} (optimized for F1-Score)\n")
            for metric, value in optimal_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        return results
        
    def compare_models(self, models_dict):
        results = {}
        
        for name, model in models_dict.items():
            self.model = model
            results[name] = self.evaluate()
            
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']
        
        comparison_data = []
        for model_name, model_results in results.items():
            model_metrics = model_results['optimal_metrics']
            row = {'model': model_name}
            row.update({metric: model_metrics[metric] for metric in metrics})
            comparison_data.append(row)
            
        comparison_df = pd.DataFrame(comparison_data)
        
        plt.figure(figsize=(12, 8))
        comparison_df.set_index('model')[metrics].plot(kind='bar', figsize=(15, 8))
        plt.title('Model Comparison - Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(axis='y')
        plt.tight_layout()
        
        timestamp = int(time.time())
        save_path = os.path.join(self.output_dir, f'model_comparison_{timestamp}.png')
        plt.savefig(save_path)
        
        return comparison_df 