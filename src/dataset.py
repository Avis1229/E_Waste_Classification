"""
E-Waste Dataset Module
Handles data loading, augmentation, and preprocessing for e-waste classification
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class EWasteDataset(Dataset):
    """
    PyTorch Dataset for E-Waste Classification
    
    Args:
        data_dir: Path to data directory (train/val/test)
        transform: Albumentations transform pipeline
        img_size: Size to resize images (default: 224)
    """
    
    def __init__(
        self, 
        data_dir: str, 
        transform: Optional[Callable] = None,
        img_size: int = 224
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Get all image paths and labels
        self.images = []
        self.labels = []
        self.classes = []
        
        # Scan directory for classes (subfolders)
        if self.data_dir.exists():
            self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            
            # Collect all images
            for class_name in self.classes:
                class_dir = self.data_dir / class_name
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.images.append(str(img_path))
                        self.labels.append(self.class_to_idx[class_name])
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = self.labels[idx]
        
        return image, label
    
    def get_class_counts(self):
        """Return dictionary of class counts for checking imbalance"""
        from collections import Counter
        return dict(Counter(self.labels))


def get_train_transforms(img_size: int = 224) -> A.Compose:
    """
    Training augmentation pipeline
    Includes various augmentations to improve model generalization
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5),
        A.GaussNoise(p=0.3),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """
    Validation/Test augmentation pipeline
    Only basic preprocessing without augmentation
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data
        test_dir: Path to test data
        batch_size: Batch size for training
        img_size: Image size for resizing
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = EWasteDataset(
        train_dir, 
        transform=get_train_transforms(img_size)
    )
    
    val_dataset = EWasteDataset(
        val_dir,
        transform=get_val_transforms(img_size)
    )
    
    test_dataset = EWasteDataset(
        test_dir,
        transform=get_val_transforms(img_size)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_names(data_dir: str) -> list:
    """
    Get list of class names from data directory
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        List of class names
    """
    data_path = Path(data_dir)
    if data_path.exists():
        return sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    return []


if __name__ == "__main__":
    # Test the dataset
    print("Testing E-Waste Dataset...")
    
    # Example usage
    train_dir = "../data/train"
    val_dir = "../data/val"
    test_dir = "../data/test"
    
    if Path(train_dir).exists():
        # Create dataset
        dataset = EWasteDataset(train_dir, transform=get_train_transforms())
        print(f"Dataset size: {len(dataset)}")
        print(f"Classes: {dataset.classes}")
        print(f"Number of classes: {len(dataset.classes)}")
        
        # Check class distribution
        class_counts = dataset.get_class_counts()
        print("\nClass distribution:")
        for class_name, count in zip(dataset.classes, class_counts.values()):
            print(f"  {class_name}: {count} images")
        
        # Test loading a sample
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"\nSample image shape: {image.shape}")
            print(f"Sample label: {label} ({dataset.classes[label]})")
    else:
        print(f"Data directory not found: {train_dir}")
        print("Please organize your data into train/val/test folders with class subfolders")
