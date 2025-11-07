"""
Dataset Splitting Utility
Automatically splits a dataset into train/val/test sets
"""

import os
import shutil
from pathlib import Path
from typing import Tuple
import random


def split_dataset(
    source_dir: str,
    dest_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split a dataset into train/val/test sets
    
    Args:
        source_dir: Source directory containing class folders with images
        dest_dir: Destination directory for split dataset
        train_ratio: Proportion of data for training (default: 0.8)
        val_ratio: Proportion of data for validation (default: 0.1)
        test_ratio: Proportion of data for testing (default: 0.1)
        seed: Random seed for reproducibility
    """
    
    # Validate ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001:
        raise ValueError("Ratios must sum to 1.0")
    
    random.seed(seed)
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    print("=" * 70)
    print("  DATASET SPLITTING UTILITY")
    print("=" * 70)
    print(f"\nSource: {source_path.absolute()}")
    print(f"Destination: {dest_path.absolute()}")
    print(f"Split: Train={train_ratio*100}%, Val={val_ratio*100}%, Test={test_ratio*100}%")
    print(f"Random seed: {seed}\n")
    
    # Get all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"❌ No class directories found in {source_dir}")
        print("\nExpected structure:")
        print("  source_dir/")
        print("    ├── class1/")
        print("    │   ├── img1.jpg")
        print("    │   └── img2.jpg")
        print("    ├── class2/")
        print("    └── ...")
        return
    
    print(f"Found {len(class_dirs)} classes:")
    for class_dir in sorted(class_dirs):
        print(f"  - {class_dir.name}")
    
    print("\n" + "-" * 70)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    # Process each class
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        print(f"\nProcessing class: {class_name}")
        
        # Get all images in class
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        images = []
        for ext in image_extensions:
            images.extend(list(class_dir.glob(f'*{ext}')))
            images.extend(list(class_dir.glob(f'*{ext.upper()}')))
        
        if not images:
            print(f"  ⚠️  No images found in {class_name}, skipping...")
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        n_test = n_images - n_train - n_val  # Remaining goes to test
        
        print(f"  Total images: {n_images}")
        print(f"    Train: {n_train}")
        print(f"    Val:   {n_val}")
        print(f"    Test:  {n_test}")
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create destination directories
        train_dir = dest_path / 'train' / class_name
        val_dir = dest_path / 'val' / class_name
        test_dir = dest_path / 'test' / class_name
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        print(f"  Copying files...")
        for img in train_images:
            shutil.copy2(img, train_dir / img.name)
        
        for img in val_images:
            shutil.copy2(img, val_dir / img.name)
        
        for img in test_images:
            shutil.copy2(img, test_dir / img.name)
        
        total_train += n_train
        total_val += n_val
        total_test += n_test
        
        print(f"  ✅ Done!")
    
    # Summary
    print("\n" + "=" * 70)
    print("  SPLIT SUMMARY")
    print("=" * 70)
    print(f"\nTotal images processed: {total_train + total_val + total_test}")
    print(f"  Training:   {total_train} ({total_train/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"  Validation: {total_val} ({total_val/(total_train+total_val+total_test)*100:.1f}%)")
    print(f"  Testing:    {total_test} ({total_test/(total_train+total_val+total_test)*100:.1f}%)")
    
    print(f"\n✅ Dataset split completed successfully!")
    print(f"📁 Output location: {dest_path.absolute()}")
    print("\nNext steps:")
    print("  1. Run validation: python src/validate_data.py")
    print("  2. Check the split ratios and class distributions")
    print("  3. Proceed to Phase 2 (Model Building)")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--source', type=str, required=True,
                        help='Source directory containing class folders')
    parser.add_argument('--dest', type=str, default='../data',
                        help='Destination directory for split dataset (default: ../data)')
    parser.add_argument('--train', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    try:
        split_dataset(
            source_dir=args.source,
            dest_dir=args.dest,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nUsage example:")
        print("  python split_dataset.py --source /path/to/raw/data --dest ../data")
