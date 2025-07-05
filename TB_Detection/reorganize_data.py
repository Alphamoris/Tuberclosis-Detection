"""
Dataset Reorganization Script for TB Detection
This script helps reorganize the dataset into the expected structure for training.
"""

import os
import shutil
import glob
import argparse
from tqdm import tqdm

def reorganize_dataset(data_dir):
    print("=" * 60)
    print("Dataset Reorganization for TB Detection")
    print("=" * 60)
    
    tb_dir = os.path.join(data_dir, 'TB')
    normal_dir = os.path.join(data_dir, 'Normal')
    
    tb_temp = os.path.join(data_dir, 'TB_temp')
    normal_temp = os.path.join(data_dir, 'Normal_temp')
    
    os.makedirs(tb_temp, exist_ok=True)
    os.makedirs(normal_temp, exist_ok=True)
    
    tb_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        tb_images.extend(glob.glob(os.path.join(tb_dir, '**', f'*{ext}'), recursive=True))
    
    print(f"Found {len(tb_images)} TB images")
    
    normal_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        normal_images.extend(glob.glob(os.path.join(normal_dir, f'*{ext}'), recursive=False))
    
    print(f"Found {len(normal_images)} Normal images")
    
    if not tb_images:
        print(f"No TB images found in {tb_dir}. Please check your directory structure.")
        return False
        
    if not normal_images:
        print(f"No Normal images found in {normal_dir}. Please check your directory structure.")
        return False
    
    print("Copying TB images...")
    for img_path in tqdm(tb_images):
        dest = os.path.join(tb_temp, os.path.basename(img_path))
        shutil.copy2(img_path, dest)
    
    print("Copying Normal images...")
    for img_path in tqdm(normal_images):
        dest = os.path.join(normal_temp, os.path.basename(img_path))
        shutil.copy2(img_path, dest)
    
    print("Replacing directories...")
    shutil.rmtree(tb_dir)
    shutil.rmtree(normal_dir)
    
    os.rename(tb_temp, tb_dir)
    os.rename(normal_temp, normal_dir)
    
    print("\nDataset successfully reorganized!")
    print(f"TB images: {len(os.listdir(tb_dir))}")
    print(f"Normal images: {len(os.listdir(normal_dir))}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize the TB Detection dataset")
    parser.add_argument("--data_dir", type=str, default="TB_Detection/data", 
                        help="Path to the data directory")
    
    args = parser.parse_args()
    reorganize_dataset(args.data_dir) 