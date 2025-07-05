import os
import argparse
import tensorflow as tf
import numpy as np
import json
import time
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

from utils.data_loader import DataLoader
from utils.preprocessor import ImagePreprocessor
from utils.augmentation import DataAugmenter
from models.model_architectures import ModelBuilder
from models.training_pipeline import TrainingPipeline
from models.callbacks import MetricsLogger, ConfusionMatrixCallback, LearningRateLogger, VisualizeActivations
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer

def train_model(args):
    print("=" * 50)
    print(f"Starting training for {args.model_name} model")
    print("=" * 50)
    
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nStep 1: Loading and preparing data...")
    data_loader = DataLoader(dataset_path=args.data_dir)
    
    if args.download_dataset:
        print("Downloading dataset from Kaggle...")
        data_loader.download_dataset()
    
    if not os.path.exists(os.path.join(args.data_dir, 'train')):
        print("Organizing dataset into train/valid/test splits...")
        tb_dir = os.path.join(args.data_dir, 'TB')
        normal_dir = os.path.join(args.data_dir, 'Normal')
        
        if os.path.exists(tb_dir) and os.path.exists(normal_dir):
            dataset_stats = data_loader.organize_data(
                tb_dir=tb_dir,
                normal_dir=normal_dir,
                test_size=args.test_size,
                valid_size=args.valid_size
            )
            
            if dataset_stats is None:
                print(f"Error: Failed to organize dataset. Please check data directory structure.")
                return None
                
            print("\nDataset statistics:")
            for split, stats in dataset_stats.items():
                print(f"  {split}: {stats}")
        else:
            print(f"Error: TB and/or Normal directories not found in {args.data_dir}")
            print(f"Current directories in {args.data_dir}: {os.listdir(args.data_dir)}")
            print("Please organize your data with TB/ and Normal/ folders containing X-ray images")
            return None
    
    print("\nCreating TensorFlow datasets...")
    train_ds, valid_ds, test_ds = data_loader.create_tf_datasets(
        img_height=args.image_size,
        img_width=args.image_size,
        batch_size=args.batch_size
    )
    
    print("\nConfiguring data augmentation...")
    augmenter = DataAugmenter(
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=(0.8, 1.2),
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.8, 1.2)
    )
    
    print("\nApplying data augmentation...")
    aug_layer = augmenter.create_tf_augmentation_layer()
    
    train_ds = train_ds.map(
        lambda x, y: (aug_layer(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    print("\nStep 2: Building model...")
    model_builder = ModelBuilder(input_shape=(args.image_size, args.image_size, 3))
    
    model = model_builder.build_model(
        model_name=args.model_name,
        trainable_base=False,
        dropout_rates=(args.dropout_rate, args.dropout_rate * 0.6)
    )
    
    print(f"Model architecture: {args.model_name}")
    
    print("\nStep 3: Training configuration...")
    class_weights = data_loader.get_class_weights() if args.use_class_weights else None
    
    if class_weights:
        print(f"Using class weights: {class_weights}")
    
    training_pipeline = TrainingPipeline(model, model_name=args.model_name, output_dir=output_dir)
    
    model = training_pipeline.compile_model(learning_rate=args.learning_rate)
    
    callbacks = training_pipeline.setup_callbacks(
        monitor='val_auc', 
        patience=args.early_stopping_patience,
        min_delta=0.001
    )
    
    callbacks.extend([
        MetricsLogger(log_dir=os.path.join(output_dir, 'logs')),
        ConfusionMatrixCallback(validation_data=valid_ds, 
                               log_dir=os.path.join(output_dir, 'logs')),
        LearningRateLogger(log_dir=os.path.join(output_dir, 'logs')),
    ])
    
    print("\nStep 4: Training model...")
    start_time = time.time()
    
    history = training_pipeline.train(
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weights=class_weights,
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    training_pipeline.plot_training_history()
    
    print("\nStep 5: Fine-tuning model...")
    if args.fine_tune:
        print("Starting fine-tuning phase...")
        
        fine_tune_history = training_pipeline.fine_tune(
            train_dataset=train_ds,
            valid_dataset=valid_ds,
            learning_rate=args.fine_tune_lr,
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            class_weights=class_weights,
            unfreeze_layers=args.unfreeze_layers
        )
        
        print("\nFine-tuning completed")
        
    print("\nStep 6: Model evaluation...")
    evaluator = ModelEvaluator(
        model=model,
        test_dataset=test_ds,
        output_dir=os.path.join(output_dir, 'evaluation')
    )
    
    evaluation_results = evaluator.evaluate()
    
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=4, cls=NumpyEncoder)
        
    print(f"\nEvaluation results saved to {results_file}")
    
    print("\nStep 7: Saving model...")
    model_save_path = training_pipeline.save_model()
    print(f"Model saved to {model_save_path}")
    
    print("\nStep 8: Visualizing results...")
    visualizer = Visualizer(output_dir=os.path.join(output_dir, 'visualizations'))
    
    images, labels, predictions = evaluator.predict_all()
    
    if len(images) >= 10:
        sample_indices = np.random.choice(len(images), size=10, replace=False)
        sample_images = images[sample_indices]
        sample_labels = labels[sample_indices]
        sample_predictions = predictions[sample_indices]
        
        for i, (img, label, pred) in enumerate(zip(sample_images, sample_labels, sample_predictions)):
            visualizer.plot_gradcam(model, img)
    
    print("\nTraining and evaluation completed successfully!")
    
    return {
        'model': model,
        'evaluation_results': evaluation_results,
        'model_path': model_save_path,
        'training_time': training_time
    }

def train_all_models(args):
    results = {}
    
    for model_name in ['resnet50', 'vgg16', 'efficientnet']:
        args.model_name = model_name
        print(f"\n\n{'='*20} Training {model_name} {'='*20}\n")
        model_results = train_model(args)
        
        if model_results is None:
            print(f"Training failed for {model_name}, skipping to next model...")
            continue
            
        results[model_name] = {
            'model_path': model_results['model_path'],
            'evaluation_results': model_results['evaluation_results'],
            'training_time': model_results['training_time']
        }
    
    if not results:
        print("\n\n" + "="*50)
        print("No models were successfully trained. Please check your dataset structure.")
        print("="*50)
        return None
        
    print("\n\n" + "="*50)
    print("All models trained successfully!")
    print("="*50)
    print("\nModel comparison:")
    
    for model_name, model_result in results.items():
        eval_results = model_result['evaluation_results']
        default_metrics = eval_results['default_metrics']
        
        print(f"\n{model_name}:")
        print(f"  Accuracy: {default_metrics['accuracy']:.4f}")
        print(f"  Precision: {default_metrics['precision']:.4f}")
        print(f"  Recall: {default_metrics['recall']:.4f}")
        print(f"  F1-Score: {default_metrics['f1_score']:.4f}")
        print(f"  AUC: {default_metrics['roc_auc']:.4f}")
        print(f"  Training time: {model_result['training_time']/60:.2f} minutes")
    
    results_file = os.path.join(args.output_dir, 'all_models_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nAll results saved to {results_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TB Detection models")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to save outputs")
    parser.add_argument("--model_name", type=str, default="resnet50", 
                        choices=["resnet50", "vgg16", "efficientnet", "all"],
                        help="Model architecture to use")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--early_stopping_patience", type=int, default=10, 
                        help="Patience for early stopping")
    parser.add_argument("--fine_tune", action="store_true", help="Whether to fine-tune the model")
    parser.add_argument("--fine_tune_epochs", type=int, default=30, 
                        help="Number of fine-tuning epochs")
    parser.add_argument("--fine_tune_lr", type=float, default=0.0001, 
                        help="Learning rate for fine-tuning")
    parser.add_argument("--unfreeze_layers", type=int, default=20, 
                        help="Number of layers to unfreeze for fine-tuning")
    parser.add_argument("--use_class_weights", action="store_true", 
                        help="Whether to use class weights for imbalanced data")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="Proportion of data to use for testing")
    parser.add_argument("--valid_size", type=float, default=0.2, 
                        help="Proportion of training data to use for validation")
    parser.add_argument("--download_dataset", action="store_true", 
                        help="Whether to download the dataset from Kaggle")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"TB Detection Model Training")
    print("="*70)
    
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    if args.model_name == "all":
        train_all_models(args)
    else:
        train_model(args) 