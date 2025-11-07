"""
Evaluation Script
Evaluate trained model with comprehensive metrics and visualizations
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, accuracy_score
)
from tqdm import tqdm

from model import create_model
from dataset import EWasteDataset, get_val_transforms


class ModelEvaluator:
    """Evaluator class for comprehensive model evaluation"""
    
    def __init__(self, model_path: str, test_dir: str, device: str = None):
        self.model_path = Path(model_path)
        self.test_dir = test_dir
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load checkpoint
        print(f"📦 Loading model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get config
        self.config = checkpoint.get('config', {})
        self.model_name = self.config.get('model_name', 'efficientnet')
        self.num_classes = self.config.get('num_classes', 8)
        
        # Create model
        self.model = create_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained=False,
            freeze_base=False
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model: {self.model_name}")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Best val acc: {checkpoint.get('best_val_acc', 'unknown'):.2f}%")
        
        # Load test dataset
        print(f"\n📁 Loading test dataset from {test_dir}...")
        self.test_dataset = EWasteDataset(
            test_dir,
            transform=get_val_transforms(self.config.get('img_size', 224))
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        self.class_names = self.test_dataset.classes
        print(f"   Test samples: {len(self.test_dataset)}")
        print(f"   Classes: {self.class_names}")
        
        # Initialize results storage
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []
    
    @torch.no_grad()
    def predict(self) -> Tuple[List, List, List]:
        """Run inference on test set"""
        print("\n🔮 Running inference on test set...")
        
        self.model.eval()
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []
        
        for images, labels in tqdm(self.test_loader, desc='Predicting'):
            images = images.to(self.device)
            
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            self.all_predictions.extend(predicted.cpu().numpy())
            self.all_labels.extend(labels.numpy())
            self.all_probs.extend(probs.cpu().numpy())
        
        self.all_predictions = np.array(self.all_predictions)
        self.all_labels = np.array(self.all_labels)
        self.all_probs = np.array(self.all_probs)
        
        return self.all_predictions, self.all_labels, self.all_probs
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        print("\n📊 Calculating metrics...")
        
        # Overall accuracy
        accuracy = accuracy_score(self.all_labels, self.all_predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_labels,
            self.all_predictions,
            average=None,
            labels=range(self.num_classes)
        )
        
        # Average metrics
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            self.all_labels,
            self.all_predictions,
            average='macro'
        )
        
        # Weighted average metrics
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            self.all_labels,
            self.all_predictions,
            average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_avg,
            'recall_macro': recall_avg,
            'f1_macro': f1_avg,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class': {
                self.class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(self.num_classes)
            }
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics"""
        print("\n" + "="*60)
        print("  EVALUATION RESULTS")
        print("="*60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']*100:.2f}%")
        print(f"  Precision (macro):  {metrics['precision_macro']*100:.2f}%")
        print(f"  Recall (macro):     {metrics['recall_macro']*100:.2f}%")
        print(f"  F1-Score (macro):   {metrics['f1_macro']*100:.2f}%")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 65)
        
        for class_name in self.class_names:
            class_metrics = metrics['per_class'][class_name]
            print(f"{class_name:<20} "
                  f"{class_metrics['precision']*100:>9.2f}% "
                  f"{class_metrics['recall']*100:>9.2f}% "
                  f"{class_metrics['f1']*100:>9.2f}% "
                  f"{class_metrics['support']:>10}")
        
        print("="*60 + "\n")
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot and save confusion matrix"""
        print("📈 Generating confusion matrix...")
        
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot with percentages
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Percentage (%)'}
        )
        
        plt.title(f'Confusion Matrix - {self.model_name.upper()}\nAccuracy: {accuracy_score(self.all_labels, self.all_predictions)*100:.2f}%',
                  fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_per_class_metrics(self, metrics: Dict, save_path: str = None):
        """Plot per-class precision, recall, and F1-score"""
        print("📈 Generating per-class metrics chart...")
        
        classes = self.class_names
        precision = [metrics['per_class'][c]['precision'] * 100 for c in classes]
        recall = [metrics['per_class'][c]['recall'] * 100 for c in classes]
        f1 = [metrics['per_class'][c]['f1'] * 100 for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
        bars2 = ax.bar(x, recall, width, label='Recall', color='coral')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='seagreen')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score (%)', fontsize=12)
        ax.set_title(f'Per-Class Metrics - {self.model_name.upper()}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def evaluate(self, output_dir: str = "../models"):
        """Run complete evaluation pipeline"""
        print("\n" + "="*60)
        print("  MODEL EVALUATION")
        print("="*60)
        
        # Run predictions
        self.predict()
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Generate visualizations
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        cm_path = output_path / f"{self.model_name}_confusion_matrix.png"
        self.plot_confusion_matrix(str(cm_path))
        
        # Per-class metrics
        metrics_path = output_path / f"{self.model_name}_per_class_metrics.png"
        self.plot_per_class_metrics(metrics, str(metrics_path))
        
        # Save metrics to JSON
        import json
        metrics_json_path = output_path / f"{self.model_name}_test_metrics.json"
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Metrics saved to {metrics_json_path}")
        
        # Generate classification report
        report = classification_report(
            self.all_labels,
            self.all_predictions,
            target_names=self.class_names,
            digits=4
        )
        
        report_path = output_path / f"{self.model_name}_classification_report.txt"
        with open(report_path, 'w') as f:
            f.write("E-WASTE CLASSIFIER - CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        print(f"📄 Classification report saved to {report_path}")
        
        print("\n" + "="*60)
        print("  EVALUATION COMPLETE ✅")
        print("="*60 + "\n")
        
        return metrics


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate e-waste classifier')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--test_dir', type=str, default='../data/test',
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    # Check if test directory exists
    if not Path(args.test_dir).exists():
        print(f"❌ Test directory not found: {args.test_dir}")
        return
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(
        model_path=args.model,
        test_dir=args.test_dir,
        device=args.device
    )
    
    metrics = evaluator.evaluate(output_dir=args.output_dir)
    
    # Check success criteria
    print("\n🎯 Checking Success Criteria:")
    print(f"   Target accuracy: >90% | Actual: {metrics['accuracy']*100:.2f}%")
    print(f"   Target F1-score: >85% | Actual: {metrics['f1_macro']*100:.2f}%")
    
    if metrics['accuracy'] > 0.90:
        print("   ✅ Accuracy target met!")
    else:
        print("   ⚠️  Accuracy target not met")
    
    if metrics['f1_macro'] > 0.85:
        print("   ✅ F1-score target met!")
    else:
        print("   ⚠️  F1-score target not met")


if __name__ == "__main__":
    main()
