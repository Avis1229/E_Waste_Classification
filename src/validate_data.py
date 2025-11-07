"""
Data Validation Script
Checks for corrupted images, class imbalance, and data quality issues
"""

import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import json

from PIL import Image
from tqdm import tqdm
import numpy as np


def check_corrupted_images(data_dir: str) -> List[str]:
    """
    Check for corrupted or unreadable images
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        List of corrupted image paths
    """
    corrupted_images = []
    data_path = Path(data_dir)
    
    print(f"\n🔍 Checking for corrupted images in {data_dir}...")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(data_path.rglob(f'*{ext}'))
        all_images.extend(data_path.rglob(f'*{ext.upper()}'))
    
    # Check each image
    for img_path in tqdm(all_images, desc="Validating images"):
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify it's a valid image
            
            # Try to actually load it
            with Image.open(img_path) as img:
                img.load()
                
        except Exception as e:
            print(f"  ❌ Corrupted: {img_path} - {str(e)}")
            corrupted_images.append(str(img_path))
    
    if corrupted_images:
        print(f"  ⚠️  Found {len(corrupted_images)} corrupted images")
    else:
        print("  ✅ No corrupted images found")
    
    return corrupted_images


def check_class_balance(data_dir: str) -> Dict[str, int]:
    """
    Check class distribution and imbalance
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        Dictionary with class counts
    """
    data_path = Path(data_dir)
    class_counts = {}
    
    print(f"\n📊 Checking class balance in {data_dir}...")
    
    # Count images per class
    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            count = 0
            for ext in image_extensions:
                count += len(list(class_dir.glob(f'*{ext}')))
                count += len(list(class_dir.glob(f'*{ext.upper()}')))
            
            class_counts[class_dir.name] = count
    
    # Display results
    if class_counts:
        total_images = sum(class_counts.values())
        print(f"  Total images: {total_images}")
        print(f"  Number of classes: {len(class_counts)}")
        print("\n  Class distribution:")
        
        max_count = max(class_counts.values()) if class_counts else 0
        min_count = min(class_counts.values()) if class_counts else 0
        
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / total_images * 100) if total_images > 0 else 0
            bar = '█' * int(percentage / 2)
            print(f"    {class_name:20s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        # Check for imbalance
        if max_count > 0 and min_count > 0:
            imbalance_ratio = max_count / min_count
            print(f"\n  Imbalance ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 3:
                print("  ⚠️  Significant class imbalance detected! Consider:")
                print("     - Collecting more data for underrepresented classes")
                print("     - Using class weights during training")
                print("     - Applying data augmentation")
            elif imbalance_ratio > 1.5:
                print("  ⚠️  Moderate class imbalance detected")
            else:
                print("  ✅ Classes are well balanced")
        
        # Check minimum samples per class
        if min_count < 100:
            print(f"  ⚠️  Some classes have fewer than 100 images!")
            print("     This may lead to poor model performance")
        elif min_count < 500:
            print(f"  ⚠️  Some classes have fewer than 500 images")
            print("     Consider collecting more data if possible")
        else:
            print(f"  ✅ All classes have sufficient samples (min: {min_count})")
    
    return class_counts


def check_image_quality(data_dir: str, sample_size: int = 100) -> Dict:
    """
    Check image quality metrics (size, resolution, channels)
    
    Args:
        data_dir: Path to data directory
        sample_size: Number of images to sample for quality check
    
    Returns:
        Dictionary with quality metrics
    """
    data_path = Path(data_dir)
    
    print(f"\n🎨 Checking image quality (sampling {sample_size} images)...")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(data_path.rglob(f'*{ext}'))
        all_images.extend(data_path.rglob(f'*{ext.upper()}'))
    
    if not all_images:
        print("  ⚠️  No images found!")
        return {}
    
    # Sample images
    sample_images = np.random.choice(all_images, min(sample_size, len(all_images)), replace=False)
    
    widths = []
    heights = []
    channels = []
    formats = []
    file_sizes = []
    
    for img_path in tqdm(sample_images, desc="Analyzing images"):
        try:
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)
                channels.append(len(img.getbands()))
                formats.append(img.format)
                file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
        except Exception as e:
            continue
    
    # Calculate statistics
    if widths:
        print(f"\n  Image dimensions:")
        print(f"    Width:  {np.mean(widths):.0f} ± {np.std(widths):.0f} px (min: {np.min(widths)}, max: {np.max(widths)})")
        print(f"    Height: {np.mean(heights):.0f} ± {np.std(heights):.0f} px (min: {np.min(heights)}, max: {np.max(heights)})")
        
        print(f"\n  File sizes:")
        print(f"    Average: {np.mean(file_sizes):.1f} KB")
        print(f"    Range: {np.min(file_sizes):.1f} - {np.max(file_sizes):.1f} KB")
        
        print(f"\n  Image channels:")
        channel_counts = Counter(channels)
        for ch, count in channel_counts.items():
            print(f"    {ch} channels: {count} images")
            if ch != 3:
                print(f"      ⚠️  Non-RGB images detected!")
        
        print(f"\n  Image formats:")
        format_counts = Counter(formats)
        for fmt, count in format_counts.items():
            print(f"    {fmt}: {count} images")
        
        # Quality warnings
        if np.min(widths) < 100 or np.min(heights) < 100:
            print("\n  ⚠️  Some images are very small (<100px)")
            print("     This may affect model performance")
        else:
            print("\n  ✅ Image sizes look good")
        
        return {
            'width_mean': float(np.mean(widths)),
            'width_std': float(np.std(widths)),
            'height_mean': float(np.mean(heights)),
            'height_std': float(np.std(heights)),
            'file_size_mean_kb': float(np.mean(file_sizes)),
            'channels': dict(channel_counts),
            'formats': dict(format_counts)
        }
    
    return {}


def validate_dataset_structure(base_dir: str) -> bool:
    """
    Validate that the dataset has the correct structure
    
    Args:
        base_dir: Base directory containing train/val/test folders
    
    Returns:
        True if structure is valid, False otherwise
    """
    base_path = Path(base_dir)
    
    print("\n🏗️  Validating dataset structure...")
    
    required_dirs = ['train', 'val', 'test']
    all_valid = True
    
    for split in required_dirs:
        split_path = base_path / split
        if not split_path.exists():
            print(f"  ❌ Missing directory: {split}")
            all_valid = False
        else:
            # Check for class subdirectories
            class_dirs = [d for d in split_path.iterdir() if d.is_dir()]
            if not class_dirs:
                print(f"  ⚠️  {split} directory exists but has no class subdirectories")
                all_valid = False
            else:
                print(f"  ✅ {split}: {len(class_dirs)} classes found")
    
    return all_valid


def run_full_validation(base_dir: str, output_file: str = "data_validation_report.json"):
    """
    Run all validation checks and save report
    
    Args:
        base_dir: Base directory containing train/val/test folders
        output_file: Path to save validation report
    """
    print("="*60)
    print("  E-WASTE DATASET VALIDATION REPORT")
    print("="*60)
    
    base_path = Path(base_dir)
    report = {
        'base_directory': str(base_path.absolute()),
        'structure_valid': False,
        'corrupted_images': {},
        'class_distribution': {},
        'quality_metrics': {}
    }
    
    # Check structure
    report['structure_valid'] = validate_dataset_structure(base_dir)
    
    # Check each split
    for split in ['train', 'val', 'test']:
        split_path = base_path / split
        if split_path.exists():
            print(f"\n{'='*60}")
            print(f"  Validating {split.upper()} set")
            print(f"{'='*60}")
            
            # Check for corrupted images
            corrupted = check_corrupted_images(str(split_path))
            report['corrupted_images'][split] = corrupted
            
            # Check class balance
            class_counts = check_class_balance(str(split_path))
            report['class_distribution'][split] = class_counts
            
            # Check image quality
            quality = check_image_quality(str(split_path))
            report['quality_metrics'][split] = quality
    
    # Save report
    report_path = base_path / output_file
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  ✅ Validation complete!")
    print(f"  📄 Report saved to: {report_path}")
    print(f"{'='*60}\n")
    
    return report


if __name__ == "__main__":
    # Run validation on the data directory
    data_dir = "../data"
    
    if Path(data_dir).exists():
        report = run_full_validation(data_dir)
    else:
        print(f"❌ Data directory not found: {data_dir}")
        print("\nPlease ensure your data is organized as follows:")
        print("  data/")
        print("  ├── train/")
        print("  │   ├── batteries/")
        print("  │   ├── cables/")
        print("  │   ├── chargers/")
        print("  │   └── ...")
        print("  ├── val/")
        print("  │   └── (same structure)")
        print("  └── test/")
        print("      └── (same structure)")
