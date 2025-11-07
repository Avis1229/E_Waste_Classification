"""
Prediction/Inference Script
Single image prediction for deployment and testing
"""

import time
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from model import create_model
from dataset import get_val_transforms


class EWastePredictor:
    """Predictor class for single image inference"""
    
    def __init__(self, model_path: str, device: str = None):
        self.model_path = Path(model_path)
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load checkpoint
        print(f"📦 Loading model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get config
        self.config = checkpoint.get('config', {})
        self.model_name = self.config.get('model_name', 'efficientnet')
        self.num_classes = self.config.get('num_classes', 8)
        self.img_size = self.config.get('img_size', 224)
        
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
        
        # Get class names from a sample data directory or use defaults
        self.class_names = self._get_class_names()
        
        # Transforms
        self.transform = get_val_transforms(self.img_size)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model: {self.model_name}")
        print(f"   Classes: {self.class_names}")
        print(f"   Device: {self.device}")
    
    def _get_class_names(self) -> List[str]:
        """Get class names from config or use defaults"""
        # Try to get from test_dir if available
        test_dir = self.config.get('test_dir', '../data/test')
        test_path = Path(test_dir)
        
        if test_path.exists():
            classes = sorted([d.name for d in test_path.iterdir() if d.is_dir()])
            if classes:
                return classes
        
        # Default class names
        return ['batteries', 'cables', 'chargers', 'circuit_boards', 
                'keyboards', 'phones', 'monitors', 'misc']
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        augmented = self.transform(image=image)
        image_tensor = augmented['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    @torch.no_grad()
    def predict(
        self, 
        image_path: str, 
        top_k: int = 3
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
        
        Returns:
            Tuple of (predicted_class, confidence, top_k_predictions)
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        start_time = time.time()
        output = self.model(image_tensor)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get probabilities
        probs = F.softmax(output, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes))
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Format results
        predicted_class = self.class_names[top_indices[0]]
        confidence = float(top_probs[0])
        
        top_k_predictions = [
            (self.class_names[idx], float(prob))
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return predicted_class, confidence, top_k_predictions, inference_time
    
    def predict_batch(
        self, 
        image_paths: List[str]
    ) -> List[Dict]:
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_path in image_paths:
            try:
                pred_class, confidence, top_k, inf_time = self.predict(image_path)
                
                results.append({
                    'image_path': image_path,
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'top_predictions': top_k,
                    'inference_time_ms': inf_time,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results


def print_prediction(
    image_path: str,
    predicted_class: str,
    confidence: float,
    top_k_predictions: List[Tuple[str, float]],
    inference_time: float
):
    """Print formatted prediction results"""
    print("\n" + "="*60)
    print("  PREDICTION RESULTS")
    print("="*60)
    print(f"\nImage: {image_path}")
    print(f"\n🎯 Predicted Class: {predicted_class.upper()}")
    print(f"   Confidence: {confidence*100:.2f}%")
    print(f"   Inference Time: {inference_time:.2f}ms")
    
    print(f"\n📊 Top Predictions:")
    for i, (class_name, prob) in enumerate(top_k_predictions, 1):
        bar = '█' * int(prob * 50)
        print(f"   {i}. {class_name:<20} {prob*100:>6.2f}% {bar}")
    
    print("="*60 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Predict e-waste class for an image')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top predictions to show')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return
    
    # Create predictor
    predictor = EWastePredictor(
        model_path=args.model,
        device=args.device
    )
    
    # Run prediction
    predicted_class, confidence, top_k, inference_time = predictor.predict(
        image_path=args.image,
        top_k=args.top_k
    )
    
    # Print results
    print_prediction(
        image_path=args.image,
        predicted_class=predicted_class,
        confidence=confidence,
        top_k_predictions=top_k,
        inference_time=inference_time
    )
    
    # Check inference time criteria
    if inference_time < 100:
        print("✅ Inference time meets target (<100ms)")
    else:
        print(f"⚠️  Inference time exceeds target: {inference_time:.2f}ms > 100ms")


if __name__ == "__main__":
    main()
