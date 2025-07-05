
"""
Training Monitor Script for TB Detection
This script helps monitor the progress of training and display model stats.
"""

import os
import json
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def list_models():
    """List all trained models in the output directory"""
    output_dir = Path("output")
    
    if not output_dir.exists():
        print("No output directory found. No models have been trained yet.")
        return
        
    model_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name != "models"]
    
    if not model_dirs:
        print("No model directories found in output/")
        return
    
    print("\n== Available Models ==")
    for model_dir in model_dirs:
        model_files = list(model_dir.glob("models/*_final.h5"))
        evaluation_files = list(model_dir.glob("evaluation_results.json"))
        
        status = "✓ Completed" if model_files else "⚠ In Progress"
        
        print(f"{model_dir.name}: {status}")
        
        if evaluation_files:
            try:
                with open(evaluation_files[0]) as f:
                    eval_data = json.load(f)
                
                metrics = eval_data.get("default_metrics", {})
                print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
                print(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")
                print(f"  F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
                print(f"  AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
            except:
                print("  Error reading evaluation results")
                
def check_logs():
    """Check training logs for progress"""
    output_dir = Path("output")
    
    if not output_dir.exists():
        print("No output directory found.")
        return
        
    model_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name != "models"]
    
    if not model_dirs:
        print("No model directories found in output/")
        return
    
    for model_dir in model_dirs:
        log_dir = model_dir / "logs"
        
        if not log_dir.exists():
            continue
            
        event_files = list(log_dir.glob("events.out.tfevents.*"))
        
        if event_files:
            print(f"\nModel: {model_dir.name}")
            print(f"  Training logs found: {len(event_files)} files")
            print(f"  Last modified: {time.ctime(event_files[0].stat().st_mtime)}")
            
def monitor_directory():
    """Monitor the output directory for changes"""
    print("Monitoring training progress (press Ctrl+C to stop)...")
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("=" * 50)
            print("TB Detection Training Monitor")
            print("=" * 50)
            print(f"Last updated: {time.ctime()}")
            
            list_models()
            check_logs()
            
            model_files = list(Path("output").glob("**/models/*.h5"))
            if model_files:
                newest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                print(f"\nNewest model file: {newest_model}")
                print(f"Size: {newest_model.stat().st_size / (1024*1024):.2f} MB")
                print(f"Last modified: {time.ctime(newest_model.stat().st_mtime)}")
            
            time.sleep(10) 
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor TB Detection training progress")
    parser.add_argument("--watch", action="store_true", help="Continuously monitor training progress")
    
    args = parser.parse_args()
    
    if args.watch:
        monitor_directory()
    else:
        print("=" * 50)
        print("TB Detection Training Status")
        print("=" * 50)
        list_models()
        check_logs() 