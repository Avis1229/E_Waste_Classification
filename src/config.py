"""
Training Configuration
Hyperparameters and settings for model training
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training e-waste classifier"""
    
    # Data paths
    data_dir: str = "./data"
    train_dir: str = "./data/train"
    val_dir: str = "./data/val"
    test_dir: str = "./data/test"
    
    # Model settings
    model_name: str = "efficientnet"  # 'baseline', 'efficientnet', 'mobilenet', 'resnet18'
    num_classes: int = 8
    pretrained: bool = True
    freeze_base: bool = True
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Image settings
    img_size: int = 224
    
    # Optimizer settings
    optimizer: str = "adam"  # 'adam', 'adamw', 'sgd'
    momentum: float = 0.9  # For SGD
    
    # Scheduler settings
    scheduler: str = "plateau"  # 'plateau', 'step', 'cosine', 'none'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_step_size: int = 10
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Model checkpointing
    save_dir: str = "../models"
    save_best_only: bool = True
    checkpoint_every: int = 5  # Save checkpoint every N epochs
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_interval: int = 10  # Log every N batches
    verbose: bool = True
    
    # Fine-tuning settings
    unfreeze_after_epoch: Optional[int] = None  # Unfreeze base model after N epochs
    fine_tune_lr: float = 1e-5  # Learning rate for fine-tuning
    
    # Class weights (for imbalanced datasets)
    use_class_weights: bool = False
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.model_name not in ['baseline', 'efficientnet', 'mobilenet', 'resnet18', 'resnet50']:
            raise ValueError(f"Invalid model_name: {self.model_name}")
        
        if self.optimizer not in ['adam', 'adamw', 'sgd']:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        
        if self.scheduler not in ['plateau', 'step', 'cosine', 'none']:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return self.__dict__.copy()
    
    def print_config(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("  TRAINING CONFIGURATION")
        print("="*60)
        print(f"\nModel Settings:")
        print(f"  Model:           {self.model_name}")
        print(f"  Num classes:     {self.num_classes}")
        print(f"  Pretrained:      {self.pretrained}")
        print(f"  Freeze base:     {self.freeze_base}")
        
        print(f"\nTraining Hyperparameters:")
        print(f"  Batch size:      {self.batch_size}")
        print(f"  Epochs:          {self.num_epochs}")
        print(f"  Learning rate:   {self.learning_rate}")
        print(f"  Optimizer:       {self.optimizer}")
        print(f"  Scheduler:       {self.scheduler}")
        
        print(f"\nEarly Stopping:")
        print(f"  Enabled:         {self.early_stopping}")
        print(f"  Patience:        {self.early_stopping_patience}")
        
        print(f"\nDevice & Performance:")
        print(f"  Device:          {self.device}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Num workers:     {self.num_workers}")
        
        print(f"\nData:")
        print(f"  Train dir:       {self.train_dir}")
        print(f"  Val dir:         {self.val_dir}")
        print(f"  Image size:      {self.img_size}x{self.img_size}")
        
        print("="*60 + "\n")


# Predefined configurations for different scenarios

def get_baseline_config() -> TrainingConfig:
    """Configuration for baseline CNN model"""
    return TrainingConfig(
        model_name="baseline",
        pretrained=False,
        freeze_base=False,
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-3
    )


def get_quick_test_config() -> TrainingConfig:
    """Configuration for quick testing (small epochs)"""
    return TrainingConfig(
        model_name="mobilenet",
        batch_size=16,
        num_epochs=5,
        early_stopping_patience=3,
        num_workers=2
    )


def get_efficientnet_config() -> TrainingConfig:
    """Configuration for EfficientNet transfer learning"""
    return TrainingConfig(
        model_name="efficientnet",
        pretrained=True,
        freeze_base=True,
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-4,
        unfreeze_after_epoch=20,
        fine_tune_lr=1e-5
    )


def get_mobilenet_config() -> TrainingConfig:
    """Configuration for MobileNet (lightweight)"""
    return TrainingConfig(
        model_name="mobilenet",
        pretrained=True,
        freeze_base=True,
        batch_size=64,
        num_epochs=40,
        learning_rate=1e-4
    )


def get_resnet18_config() -> TrainingConfig:
    """Configuration for ResNet18"""
    return TrainingConfig(
        model_name="resnet18",
        pretrained=True,
        freeze_base=True,
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-4,
        unfreeze_after_epoch=15
    )


if __name__ == "__main__":
    # Test configurations
    print("Testing configuration presets...\n")
    
    configs = {
        "Default": TrainingConfig(),
        "Baseline": get_baseline_config(),
        "Quick Test": get_quick_test_config(),
        "EfficientNet": get_efficientnet_config(),
        "MobileNet": get_mobilenet_config(),
        "ResNet18": get_resnet18_config()
    }
    
    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"  {name} Configuration")
        print(f"{'='*60}")
        config.print_config()
