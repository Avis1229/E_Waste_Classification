"""
Inference utilities for Streamlit app
"""
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model import create_model
from dataset import get_val_transforms
import albumentations as A


class EWasteClassifier:
    """E-waste classification model wrapper for inference"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize classifier
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use (cuda/cpu)
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_path = Path(model_path)
        
        # Load checkpoint (always load to CPU to avoid GPU device mismatches)
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        
        # Get config
        config = checkpoint.get('config', {})
        self.model_name = config.get('model_name', 'resnet50')
        self.num_classes = config.get('num_classes', 8)
        
        # Get class names
        self.class_names = checkpoint.get('class_names', [
            'Keyboards', 'Mobile', 'Mouses', 'TV', 
            'camera', 'laptop', 'microwave', 'smartwatch'
        ])
        
        # Create model
        self.model = create_model(
            model_name=self.model_name,

            num_classes=self.num_classes,
            pretrained=False
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Get transforms
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"✅ Model loaded: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Classes: {len(self.class_names)}")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image for inference
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy
        img_array = np.array(image)
        
        # Apply transforms
        transformed = self.transform(image=img_array)
        img_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def predict(self, image: Image.Image, top_k: int = 3):
        """
        Predict class for image
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            dict with predictions, probabilities, and inference time
        """
        start_time = time.time()
        
        # Preprocess
        img_tensor = self.preprocess_image(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes))
        
        inference_time = time.time() - start_time
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                'class': self.class_names[idx],
                'probability': float(prob),
                'confidence': float(prob) * 100
            })
        
        return {
            'predictions': predictions,
            'top_class': predictions[0]['class'],
            'top_confidence': predictions[0]['confidence'],
            'inference_time_ms': inference_time * 1000,
            'all_probabilities': {
                self.class_names[i]: float(probabilities[0][i]) 
                for i in range(self.num_classes)
            }
        }
    
    def predict_batch(self, images: list):
        """
        Predict classes for multiple images
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of prediction dicts
        """
        results = []
        for img in images:
            result = self.predict(img)
            results.append(result)
        return results


def get_recycling_tips(class_name: str) -> dict:
    """
    Get recycling tips for detected e-waste category
    
    Args:
        class_name: Detected class name
        
    Returns:
        dict with recycling information
    """
    tips = {
        'Keyboards': {
            'description': 'Computer keyboards contain plastic, metal, and electronic components.',
            'tips': [
                '♻️ Remove batteries if wireless',
                '🔧 Separate keycaps from base if possible',
                '📦 Take to e-waste recycling center',
                '💡 Consider donating if still functional'
            ],
            'hazards': 'Contains small electronic components and plastics',
            'recyclable': True
        },
        'Mobile': {
            'description': 'Mobile phones contain valuable materials like gold, silver, and rare earth metals.',
            'tips': [
                '🔋 Remove SIM card and memory card',
                '🔒 Factory reset to erase data',
                '📱 Take to certified e-waste recycler',
                '♻️ Many retailers offer trade-in programs',
                '💰 Some components can be refurbished'
            ],
            'hazards': 'Contains lithium battery - do not throw in regular trash!',
            'recyclable': True
        },
        'Mouses': {
            'description': 'Computer mice contain plastic housing and electronic sensors.',
            'tips': [
                '🔋 Remove batteries if wireless',
                '♻️ Take to e-waste collection point',
                '🔧 Some parts can be reused for repairs'
            ],
            'hazards': 'Contains small electronic components',
            'recyclable': True
        },
        'TV': {
            'description': 'TVs contain hazardous materials like lead, mercury, and cadmium.',
            'tips': [
                '⚠️ Never throw in regular trash!',
                '📺 Contact manufacturer for take-back program',
                '🏢 Schedule pickup with certified recycler',
                '💡 Older CRT TVs need special handling',
                '♻️ LCD/LED TVs contain recyclable materials'
            ],
            'hazards': 'Contains toxic heavy metals and phosphorus',
            'recyclable': True
        },
        'camera': {
            'description': 'Digital cameras contain batteries, circuit boards, and lens assemblies.',
            'tips': [
                '🔋 Remove all batteries',
                '💾 Remove memory cards',
                '📸 Consider donating if functional',
                '♻️ Take to electronics recycler',
                '🔧 Lens and sensors can be reused'
            ],
            'hazards': 'Contains lithium batteries and electronic components',
            'recyclable': True
        },
        'laptop': {
            'description': 'Laptops contain valuable metals, circuit boards, and rechargeable batteries.',
            'tips': [
                '💽 Remove hard drive and destroy (data security)',
                '🔋 Battery must be recycled separately',
                '♻️ Take to certified e-waste recycler',
                '💻 Consider refurbishment or donation',
                '🔒 Wipe all data before recycling'
            ],
            'hazards': 'Contains lithium battery, heavy metals, and toxic materials',
            'recyclable': True
        },
        'microwave': {
            'description': 'Microwaves contain metal, glass, and electronic components.',
            'tips': [
                '⚡ Unplug and ensure capacitor is discharged',
                '🔧 Can be disassembled for parts',
                '♻️ Take to appliance recycling center',
                '⚠️ Do not attempt to repair if not qualified',
                '🏢 Some retailers accept old appliances'
            ],
            'hazards': 'Contains high-voltage capacitor - can shock even when unplugged!',
            'recyclable': True
        },
        'smartwatch': {
            'description': 'Smartwatches contain batteries, circuit boards, and sensors.',
            'tips': [
                '🔋 Contains rechargeable battery',
                '🔒 Unpair and reset device',
                '♻️ Return to manufacturer if possible',
                '📱 Some retailers have trade-in programs',
                '💰 May contain valuable materials'
            ],
            'hazards': 'Contains lithium battery',
            'recyclable': True
        }
    }
    
    return tips.get(class_name, {
        'description': 'Electronic waste item',
        'tips': ['♻️ Take to certified e-waste recycling center'],
        'hazards': 'May contain hazardous materials',
        'recyclable': True
    })
