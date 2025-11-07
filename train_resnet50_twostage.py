"""
Two-Stage ResNet50 Training Script
Stage 1: Train with frozen base layers for 20 epochs (transfer learning)
Stage 2: Fine-tune entire network for 30 epochs with lower learning rate
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from train import Trainer
from config import TrainingConfig

if __name__ == '__main__':
    print("="*70)
    print("  TWO-STAGE RESNET50 TRAINING")
    print("="*70)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")

    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("   ⚠️  WARNING: Running on CPU. Training will be much slower.")

    print("\n" + "="*70)
    print("  STAGE 1: TRANSFER LEARNING (20 EPOCHS)")
    print("  - Freeze ResNet50 base layers")
    print("  - Train only classifier head")
    print("  - Learning rate: 1e-3")
    print("="*70)

    # Stage 1 Configuration
    stage1_config = TrainingConfig(
    # Data paths
    train_dir="./data/train",
    val_dir="./data/val",
    test_dir="./data/test",
    
    # Model settings
    model_name="resnet50",
    num_classes=8,
    pretrained=True,          # Use ImageNet pretrained weights
    freeze_base=True,         # Freeze base layers (transfer learning)
    
    # Training hyperparameters - MAXIMIZE GPU USAGE
    batch_size=64 if device.type == 'cuda' else 16,  # Increased from 32 to 64
    num_epochs=20,
    learning_rate=1e-3,       # Higher LR for training classifier head
    weight_decay=1e-4,
    
    # Image settings
    img_size=224,
    
    # Optimizer settings
    optimizer="adam",
    
    # Scheduler settings
    scheduler="plateau",
    scheduler_patience=5,
    scheduler_factor=0.5,
    
    # Early stopping
    early_stopping=True,
    early_stopping_patience=8,
    early_stopping_min_delta=0.001,
    
    # Model checkpointing
    save_dir="./models",
    save_best_only=False,     # Save checkpoints for stage 2
    checkpoint_every=5,
    
    # Data loading - MAXIMIZE GPU USAGE
    num_workers=8 if device.type == 'cuda' else 2,  # More workers for GPU
    pin_memory=True if device.type == 'cuda' else False,
    
    # Mixed precision training
    use_amp=False if device.type == 'cpu' else True,
    
    # Device
    device=str(device),
    
    # Logging
    log_interval=10,
    verbose=True,
    
    # Random seed
    seed=42
    )

    print("\n📋 Stage 1 Configuration:")
    stage1_config.print_config()

    print("\n🚀 Starting Stage 1 training...")
    print("   Expected time: 15-25 minutes on GPU\n")

    try:
        # Stage 1: Train with frozen base
        trainer_stage1 = Trainer(stage1_config)
        history_stage1 = trainer_stage1.train()
        
        stage1_best_acc = trainer_stage1.best_val_acc
        stage1_best_loss = trainer_stage1.best_val_loss
        
        print("\n" + "="*70)
        print("  ✅ STAGE 1 COMPLETED!")
        print("="*70)
        print(f"Best validation accuracy: {stage1_best_acc:.2f}%")
        print(f"Best validation loss: {stage1_best_loss:.4f}")
        print(f"Model saved to: models/resnet50_best.pth")
        
        # Ask user if they want to continue to stage 2
        print("\n" + "="*70)
        print("  STAGE 2: FINE-TUNING (30 EPOCHS)")
        print("  - Unfreeze all ResNet50 layers")
        print("  - Train entire network")
        print("  - Learning rate: 1e-4 (10x smaller)")
        print("="*70)
        
        response = input("\nContinue to Stage 2 fine-tuning? (y/n): ")
        
        if response.lower() != 'y':
            print("\n⏹️  Training stopped after Stage 1")
            print("   You can resume Stage 2 later by running this script again")
            sys.exit(0)
        
        print("\n🚀 Starting Stage 2 fine-tuning...")
        print("   Expected time: 25-35 minutes on GPU\n")
        
        # Stage 2 Configuration (Fine-tuning)
        stage2_config = TrainingConfig(
            # Data paths
            train_dir="./data/train",
            val_dir="./data/val",
            test_dir="./data/test",
            
            # Model settings
            model_name="resnet50",
            num_classes=8,
            pretrained=True,
            freeze_base=False,        # Unfreeze for fine-tuning
            
            # Training hyperparameters
            batch_size=64 if device.type == 'cuda' else 16,  # MAXIMIZE GPU USAGE
            num_epochs=30,
            learning_rate=1e-4,       # 10x smaller LR for fine-tuning
            weight_decay=1e-4,
            
            # Image settings
            img_size=224,
            
            # Optimizer settings
            optimizer="adam",
            
            # Scheduler settings
            scheduler="plateau",
            scheduler_patience=7,
            scheduler_factor=0.5,
            
            # Early stopping
            early_stopping=True,
            early_stopping_patience=10,
            early_stopping_min_delta=0.0005,
            
            # Model checkpointing
            save_dir="./models",
            save_best_only=False,
            checkpoint_every=5,
            
            # Data loading
            num_workers=2 if device.type == 'cpu' else 4,
            pin_memory=False if device.type == 'cpu' else True,
            
            # Mixed precision training
            use_amp=False if device.type == 'cpu' else True,
            
            # Device
            device=str(device),
            
            # Logging
            log_interval=10,
            verbose=True,
            
            # Random seed
            seed=42
        )
        
        print("\n📋 Stage 2 Configuration:")
        stage2_config.print_config()
        
        # Load the best model from Stage 1
        print(f"\n📦 Loading Stage 1 best model...")
        checkpoint = torch.load("./models/resnet50_best.pth", map_location=device)
        
        # Create Stage 2 trainer
        trainer_stage2 = Trainer(stage2_config)
        
        # Load Stage 1 weights
        trainer_stage2.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Unfreeze the base model
        print("🔓 Unfreezing all layers for fine-tuning...")
        trainer_stage2.model.unfreeze_base()
        
        # Update optimizer with fine-tuning parameters
        trainer_stage2.optimizer = torch.optim.Adam(
            trainer_stage2.model.parameters(),
            lr=stage2_config.learning_rate,
            weight_decay=stage2_config.weight_decay
        )
        
        # Recreate scheduler
        trainer_stage2.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer_stage2.optimizer,
            mode='min',
            patience=stage2_config.scheduler_patience,
            factor=stage2_config.scheduler_factor
        )
    
        # Reset best metrics for stage 2
        trainer_stage2.best_val_acc = stage1_best_acc
        trainer_stage2.best_val_loss = stage1_best_loss
        
        print(f"✅ Loaded! Starting accuracy: {stage1_best_acc:.2f}%\n")
        
        # Stage 2: Fine-tune entire network
        history_stage2 = trainer_stage2.train()
        
        stage2_best_acc = trainer_stage2.best_val_acc
        stage2_best_loss = trainer_stage2.best_val_loss
        
        print("\n" + "="*70)
        print("  ✅ STAGE 2 COMPLETED!")
        print("="*70)
        print(f"Stage 1 best accuracy: {stage1_best_acc:.2f}%")
        print(f"Stage 2 best accuracy: {stage2_best_acc:.2f}%")
        print(f"Improvement: {stage2_best_acc - stage1_best_acc:+.2f}%")
        print(f"\nFinal model saved to: models/resnet50_best.pth")
        
        print("\n" + "="*70)
        print("  🎉 TWO-STAGE TRAINING COMPLETED!")
        print("="*70)
        print(f"\n📊 Training Summary:")
        print(f"   Total epochs: 50 (20 + 30)")
        print(f"   Final validation accuracy: {stage2_best_acc:.2f}%")
        print(f"   Final validation loss: {stage2_best_loss:.4f}")
        
        print("\n📊 Next steps:")
        print("   1. Evaluate on test set:")
        print("      python src/evaluate.py --model models/resnet50_best.pth --test_dir data/test")
        print("\n   2. Test prediction:")
        print("      python src/predict.py --model models/resnet50_best.pth --image <path_to_image>")
        
        print("\n" + "="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
        print("   Latest checkpoint saved")
        
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
