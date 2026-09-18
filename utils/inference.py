"""
Inference utilities for Streamlit app
"""
import time
import torch
import torchvision.models as models
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import albumentations as A


class EWasteClassifier:
    """E-waste classification model wrapper for inference"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize classifier
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # प्रोजेक्ट के लिए कॉन्फ़िगरेशन सेटिंग्स
        self.model_name = 'resnet50'
        self.num_classes = 8
        
        # क्लासेस के नाम
        self.class_names = [
            'Keyboards', 'Mobile', 'Mouses', 'TV', 
            'camera', 'laptop', 'microwave', 'smartwatch'
        ]
        
        # --- यहाँ बदलाव किया गया है: इंटरनेट से सीधे PyTorch Hub से मॉडल लोड करना ---
        print("📥 PyTorch सर्वर से मॉडल लोड हो रहा है...")
        # यह सीधा बिना किसी लोकल फाइल के असली मॉडल उठा लेगा
        self.model = models.resnet50(pretrained=True)
        
        # इसके आखरी हिस्से (Final Layer) को आपके 8 क्लासेस के लिए सेट करना
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, self.num_classes)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        # -------------------------------------------------------------------------
        
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
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        transformed = self.transform(image=img_array)
        img_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
    
    def predict(self, image: Image.Image, top_k: int = 3):
        """
        Predict class for image
        """
        start_time = time.time()
        img_tensor = self.preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes))
        inference_time = time.time() - start_time
        
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
        """
        results = []
        for img in images:
            result = self.predict(img)
            results.append(result)
        return results


def get_recycling_tips(class_name: str) -> dict:
    """
    Get recycling tips for detected e-waste category
    """
    tips = {
        'Keyboards': {
            'description': 'Computer keyboards contain plastic, metal, and electronic components.',
            'tips': ['♻️ Remove batteries if wireless', '🔧 Separate keycaps from base if possible', '📦 Take to e-waste recycling center', '💡 Consider donating if still functional'],
            'hazards': 'Contains small electronic components and plastics',
            'recyclable': True
        },
        'Mobile': {
            'description': 'Mobile phones contain valuable materials like gold, silver, and rare earth metals.',
            'tips': ['🔋 Remove SIM card and memory card', '🔒 Factory reset to erase data', '📱 Take to certified e-waste recycler', '♻️ Many retailers offer trade-in programs', '💰 Some components can be refurbished'],
            'hazards': 'Contains lithium battery - do not throw in regular trash!',
            'recyclable': True
        },
        'Mouses': {
            'description': 'Computer mice contain plastic housing and electronic sensors.',
            'tips': ['🔋 Remove batteries if wireless', '♻️ Take to e-waste collection point', '🔧 Some parts can be reused for repairs'],
            'hazards': 'Contains small electronic components',
            'recyclable': True
        },
        'TV': {
            'description': 'TVs contain hazardous materials like lead, mercury, and cadmium.',
            'tips': ['⚠️ Never throw in regular trash!', '📺 Contact manufacturer for take-back program', '🏢 Schedule pickup with certified recycler', '💡 Older CRT TVs need special handling', '♻️ LCD/LED TVs contain recyclable materials'],
            'hazards': 'Contains toxic heavy metals and phosphorus',
            'recyclable': True
        },
        'camera': {
            'description': 'Digital cameras contain batteries, circuit boards, and lens assemblies.',
            'tips': ['🔋 Remove all batteries', '💾 Remove memory cards', '📸 Consider donating if functional', '♻️ Take to electronics recycler', '🔧 Lens and sensors can be reused'],
            'hazards': 'Contains lithium batteries and electronic components',
            'recyclable': True
        },
        'laptop': {
            'description': 'Laptops contain valuable metals, circuit boards, and rechargeable batteries.',
            'tips': ['💽 Remove hard drive and destroy (data security)', '🔋 Battery must be recycled separately', '♻️ Take to certified e-waste recycler', '💻 Consider refurbishment or donation', '🔒 Wipe all data before recycling'],
            'hazards': 'Contains lithium battery, heavy metals, and toxic materials',
            'recyclable': True
        },
        'microwave': {
            'description': 'Microwave ovens contain high-voltage transformers and heavy metals.',
            'tips': ['⚠️ High voltage hazard! Never open the case.', '🍳 Take to large appliance collection centers.', '♻️ Metals can be recovered easily.'],
            'hazards': 'High-voltage capacitors and heavy metal parts.',
            'recyclable': True
        },
        'smartwatch': {
            'description': 'Smartwatches contain dense electronics and small lithium batteries.',
            'tips': ['🔋 Completely discharge before drop-off if possible.', '⌚ Look for specific smartwatch recycling bins.', '♻️ Plastic or metal bands can often be recycled separately.'],
            'hazards': 'Small lithium-ion cells.',
            'recyclable': True
        }
    }
    return tips.get(class_name, {
        'description': 'Electronic waste item requiring special handling.',
        'tips': ['♻️ Drop off at authorized e-waste centers.', '🔒 Wipe any data if applicable.'],
        'hazards': 'May contain components unsafe for regular trash landfills.',
        'recyclable': True
    })
