import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import Metric
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class MedicalMetrics:
    @staticmethod
    def sensitivity(y_true, y_pred):
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
        return true_positives / (possible_positives + tf.keras.backend.epsilon())
        
    @staticmethod
    def specificity(y_true, y_pred):
        true_negatives = tf.reduce_sum(tf.round(tf.clip_by_value((1-y_true) * (1-y_pred), 0, 1)))
        possible_negatives = tf.reduce_sum(tf.round(tf.clip_by_value(1-y_true, 0, 1)))
        return true_negatives / (possible_negatives + tf.keras.backend.epsilon())
        
    @staticmethod
    def precision(y_true, y_pred):
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
        return true_positives / (predicted_positives + tf.keras.backend.epsilon())
        
    @staticmethod
    def negative_predictive_value(y_true, y_pred):
        true_negatives = tf.reduce_sum(tf.round(tf.clip_by_value((1-y_true) * (1-y_pred), 0, 1)))
        predicted_negatives = tf.reduce_sum(tf.round(tf.clip_by_value(1-y_pred, 0, 1)))
        return true_negatives / (predicted_negatives + tf.keras.backend.epsilon())
        
    @staticmethod
    def f1_score(y_true, y_pred):
        precision = MedicalMetrics.precision(y_true, y_pred)
        recall = MedicalMetrics.sensitivity(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
        
    @staticmethod
    def diagnostic_odds_ratio(y_true, y_pred):
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        false_positives = tf.reduce_sum(tf.round(tf.clip_by_value((1-y_true) * y_pred, 0, 1)))
        false_negatives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * (1-y_pred), 0, 1)))
        true_negatives = tf.reduce_sum(tf.round(tf.clip_by_value((1-y_true) * (1-y_pred), 0, 1)))
        
        numerator = true_positives * true_negatives
        denominator = false_positives * false_negatives
        
        return numerator / (denominator + tf.keras.backend.epsilon())

class SpecificityMetric(Metric):
    def __init__(self, name='specificity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        
        true_negatives = tf.logical_and(tf.logical_not(y_true), tf.logical_not(y_pred))
        false_positives = tf.logical_and(tf.logical_not(y_true), y_pred)
        
        true_negatives = tf.cast(true_negatives, self.dtype)
        false_positives = tf.cast(false_positives, self.dtype)
        
        if sample_weight is not None:
            true_negatives = tf.multiply(true_negatives, sample_weight)
            false_positives = tf.multiply(false_positives, sample_weight)
            
        self.true_negatives.assign_add(tf.reduce_sum(true_negatives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))
        
    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_positives + tf.keras.backend.epsilon())
        
    def reset_state(self):
        self.true_negatives.assign(0)
        self.false_positives.assign(0)

class F1ScoreMetric(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
        
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def get_custom_metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        SpecificityMetric(name='specificity'),
        F1ScoreMetric(name='f1_score')
    ]
    
def calculate_metrics_numpy(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(np.int32)
    
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except:
        roc_auc = 0.5
        
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(recall_curve, precision_curve)
    except:
        pr_auc = 0.5
        
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    dor = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'npv': npv,
        'dor': dor,
        'confusion_matrix': {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    } 