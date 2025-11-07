"""
Model Architecture Module
Contains baseline CNN and transfer learning models for e-waste classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class BaselineCNN(nn.Module):
    """
    Simple baseline CNN model for e-waste classification
    3-layer convolutional network with batch normalization and dropout
    """
    
    def __init__(self, num_classes: int = 8, dropout: float = 0.5):
        super(BaselineCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        # Input: 224x224 -> after 3 pools: 28x28 -> 128 * 28 * 28 = 100352
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNetB0 transfer learning model
    Pretrained on ImageNet with custom classifier head
    """
    
    def __init__(self, num_classes: int = 8, pretrained: bool = True, freeze_base: bool = True):
        super(EfficientNetB0Classifier, self).__init__()
        
        # Load pretrained EfficientNetB0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.base_model = models.efficientnet_b0(weights=weights)
        else:
            self.base_model = models.efficientnet_b0(weights=None)
        
        # Freeze base layers if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Get number of input features for classifier
        in_features = self.base_model.classifier[1].in_features
        
        # Replace classifier head
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)
    
    def unfreeze_base(self):
        """Unfreeze base model for fine-tuning"""
        for param in self.base_model.parameters():
            param.requires_grad = True


class MobileNetV3Classifier(nn.Module):
    """
    MobileNetV3-Small transfer learning model
    Lightweight model suitable for deployment
    """
    
    def __init__(self, num_classes: int = 8, pretrained: bool = True, freeze_base: bool = True):
        super(MobileNetV3Classifier, self).__init__()
        
        # Load pretrained MobileNetV3-Small
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.base_model = models.mobilenet_v3_small(weights=weights)
        else:
            self.base_model = models.mobilenet_v3_small(weights=None)
        
        # Freeze base layers if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Get number of input features for classifier
        in_features = self.base_model.classifier[0].in_features
        
        # Replace classifier head
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)
    
    def unfreeze_base(self):
        """Unfreeze base model for fine-tuning"""
        for param in self.base_model.parameters():
            param.requires_grad = True


class ResNet18Classifier(nn.Module):
    """
    ResNet18 transfer learning model
    Good balance between accuracy and speed
    """
    
    def __init__(self, num_classes: int = 8, pretrained: bool = True, freeze_base: bool = True):
        super(ResNet18Classifier, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.base_model = models.resnet18(weights=weights)
        else:
            self.base_model = models.resnet18(weights=None)
        
        # Freeze base layers if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Get number of input features for classifier
        in_features = self.base_model.fc.in_features
        
        # Replace classifier head
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)
    
    def unfreeze_base(self):
        """Unfreeze base model for fine-tuning"""
        for param in self.base_model.parameters():
            param.requires_grad = True


class ResNet50Classifier(nn.Module):
    """
    ResNet50 transfer learning model
    Deeper and more accurate than ResNet18, great for two-stage training
    """
    
    def __init__(self, num_classes: int = 8, pretrained: bool = True, freeze_base: bool = True):
        super(ResNet50Classifier, self).__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.base_model = models.resnet50(weights=weights)
        else:
            self.base_model = models.resnet50(weights=None)
        
        # Freeze base layers if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Get number of input features for classifier
        in_features = self.base_model.fc.in_features
        
        # Replace classifier head
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)
    
    def unfreeze_base(self):
        """Unfreeze base model for fine-tuning"""
        for param in self.base_model.parameters():
            param.requires_grad = True


def create_model(
    model_name: str = "efficientnet",
    num_classes: int = 8,
    pretrained: bool = True,
    freeze_base: bool = True
) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: Model architecture ('baseline', 'efficientnet', 'mobilenet', 'resnet18', 'resnet50')
        num_classes: Number of output classes
        pretrained: Use pretrained weights (for transfer learning models)
        freeze_base: Freeze base layers (for transfer learning)
    
    Returns:
        PyTorch model
    """
    
    models_dict = {
        'baseline': BaselineCNN,
        'efficientnet': EfficientNetB0Classifier,
        'mobilenet': MobileNetV3Classifier,
        'resnet18': ResNet18Classifier,
        'resnet50': ResNet50Classifier
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(models_dict.keys())}")
    
    if model_name == 'baseline':
        return models_dict[model_name](num_classes=num_classes)
    else:
        return models_dict[model_name](
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_base=freeze_base
        )


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'total_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """
    Print model architecture summary
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    """
    params = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"  {model_name} Summary")
    print(f"{'='*60}")
    print(f"Total parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters:    {params['frozen']:,}")
    print(f"Model size:           {params['total_mb']:.2f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test all models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    models_to_test = ['baseline', 'efficientnet', 'mobilenet', 'resnet18', 'resnet50']
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"  Testing {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = create_model(
                model_name=model_name,
                num_classes=8,
                pretrained=(model_name != 'baseline'),
                freeze_base=True
            ).to(device)
            
            # Print summary
            print_model_summary(model, model_name)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            
            print(f"Input shape:  {test_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"✅ Forward pass successful!")
            
            # Test with unfrozen base (if applicable)
            if model_name != 'baseline' and hasattr(model, 'unfreeze_base'):
                model.unfreeze_base()
                params = count_parameters(model)
                print(f"\nAfter unfreezing:")
                print(f"  Trainable parameters: {params['trainable']:,}")
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
    
    print(f"\n{'='*60}")
    print("  Model testing complete!")
    print(f"{'='*60}\n")
