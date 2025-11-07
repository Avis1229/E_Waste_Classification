"""
Training Script
Main training loop for e-waste classifier with early stopping and checkpointing
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Tuple
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import TrainingConfig, get_efficientnet_config
from model import create_model, print_model_summary
from dataset import create_dataloaders


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """Trainer class for e-waste classifier"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # Create model
        print("\n📦 Creating model...")
        self.model = create_model(
            model_name=config.model_name,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            freeze_base=config.freeze_base
        ).to(self.device)
        
        print_model_summary(self.model, config.model_name)
        
        # Create dataloaders
        print("📁 Loading datasets...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_dir=config.train_dir,
            val_dir=config.val_dir,
            test_dir=config.test_dir,
            batch_size=config.batch_size,
            img_size=config.img_size,
            num_workers=config.num_workers
        )
        
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples:   {len(self.val_loader.dataset)}")
        print(f"  Test samples:  {len(self.test_loader.dataset)}")
        
        # Loss function
        if config.use_class_weights:
            # Calculate class weights
            class_counts = self.train_loader.dataset.get_class_counts()
            weights = self._calculate_class_weights(class_counts)
            self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
            print(f"\n⚖️  Using class weights: {weights.tolist()}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            verbose=config.verbose
        ) if config.early_stopping else None
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    def _calculate_class_weights(self, class_counts: Dict) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        counts = torch.tensor(list(class_counts.values()), dtype=torch.float32)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(weights)
        return weights
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        if self.config.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on config"""
        if self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.scheduler_patience,
                factor=self.config.scheduler_factor
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_factor
            )
        elif self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.val_loader, desc='Validating', leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        }
        
        # Save latest checkpoint
        latest_path = Path(self.config.save_dir) / f"{self.config.model_name}_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.save_dir) / f"{self.config.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model to {best_path}")
        
        # Save periodic checkpoint
        if epoch % self.config.checkpoint_every == 0:
            epoch_path = Path(self.config.save_dir) / f"{self.config.model_name}_epoch{epoch}.pth"
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("  STARTING TRAINING")
        print("="*60)
        
        self.config.print_config()
        
        start_time = time.time()
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f'\nEpoch {epoch}/{self.config.num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
            print(f'  LR: {current_lr:.6f}')
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                print(f'  🎉 New best validation accuracy: {val_acc:.2f}%')
            
            # Save checkpoint
            if self.config.save_best_only:
                if is_best:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Unfreeze base model if specified
            if (self.config.unfreeze_after_epoch is not None and 
                epoch == self.config.unfreeze_after_epoch and
                hasattr(self.model, 'unfreeze_base')):
                print(f"\n🔓 Unfreezing base model for fine-tuning...")
                self.model.unfreeze_base()
                # Update optimizer with fine-tuning learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.fine_tune_lr
                print(f"   New learning rate: {self.config.fine_tune_lr}")
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
                    break
        
        # Training complete
        training_time = time.time() - start_time
        print("\n" + "="*60)
        print("  TRAINING COMPLETE")
        print("="*60)
        print(f"Training time: {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save training history
        history_path = Path(self.config.save_dir) / f"{self.config.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n📊 Training history saved to {history_path}")
        
        return self.history


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train e-waste classifier')
    parser.add_argument('--model', type=str, default='efficientnet',
                        choices=['baseline', 'efficientnet', 'mobilenet', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--config', type=str, choices=['baseline', 'quick', 'efficientnet', 'mobilenet', 'resnet18'],
                        help='Use predefined config')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        if args.config == 'baseline':
            from config import get_baseline_config
            config = get_baseline_config()
        elif args.config == 'quick':
            from config import get_quick_test_config
            config = get_quick_test_config()
        elif args.config == 'efficientnet':
            config = get_efficientnet_config()
        elif args.config == 'mobilenet':
            from config import get_mobilenet_config
            config = get_mobilenet_config()
        elif args.config == 'resnet18':
            from config import get_resnet18_config
            config = get_resnet18_config()
    else:
        config = TrainingConfig(
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    
    # Create trainer and train
    trainer = Trainer(config)
    history = trainer.train()
    
    print("\n✅ Training finished successfully!")


if __name__ == "__main__":
    main()
