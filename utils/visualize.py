"""
Visualization utilities for Streamlit app
"""
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List


def plot_confidence_bar(predictions: List[Dict], top_k: int = 3):
    """
    Create horizontal bar chart for top-k predictions
    
    Args:
        predictions: List of prediction dicts
        top_k: Number of predictions to show
        
    Returns:
        Plotly figure
    """
    preds = predictions[:top_k]
    
    classes = [p['class'] for p in preds]
    confidences = [p['confidence'] for p in preds]
    
    # Color scale: green for high confidence, yellow for medium, red for low
    colors = ['#2ecc71' if c > 70 else '#f39c12' if c > 40 else '#e74c3c' 
              for c in confidences]
    
    fig = go.Figure(go.Bar(
        y=classes,
        x=confidences,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{c:.1f}%' for c in confidences],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f'Top {top_k} Predictions',
        xaxis_title='Confidence (%)',
        yaxis_title='Class',
        height=200 + (top_k * 40),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def plot_all_probabilities(probabilities: Dict[str, float]):
    """
    Create pie chart or bar chart for all class probabilities
    
    Args:
        probabilities: Dict mapping class names to probabilities
        
    Returns:
        Plotly figure
    """
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes]
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    classes = [classes[i] for i in sorted_indices]
    probs = [probs[i] for i in sorted_indices]
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probs,
            marker=dict(
                color=probs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence %")
            ),
            text=[f'{p:.1f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='All Class Probabilities',
        xaxis_title='Class',
        yaxis_title='Probability (%)',
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    """
    Create interactive confusion matrix heatmap
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=600,
        width=700,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig


def plot_class_accuracy(accuracies: Dict[str, float]):
    """
    Create bar chart for per-class accuracy
    
    Args:
        accuracies: Dict mapping class names to accuracy
        
    Returns:
        Plotly figure
    """
    classes = list(accuracies.keys())
    accs = list(accuracies.values())
    
    colors = ['#2ecc71' if a > 0.9 else '#f39c12' if a > 0.7 else '#e74c3c' 
              for a in accs]
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=[a * 100 for a in accs],
            marker=dict(color=colors),
            text=[f'{a*100:.1f}%' for a in accs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Per-Class Accuracy',
        xaxis_title='Class',
        yaxis_title='Accuracy (%)',
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig


def plot_training_history(history: Dict):
    """
    Plot training history (loss and accuracy curves)
    
    Args:
        history: Dict with train/val loss and accuracy
        
    Returns:
        Plotly figure
    """
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss', 'Accuracy')
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_loss'], 
                   name='Train Loss', line=dict(color='#3498db')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], 
                   name='Val Loss', line=dict(color='#e74c3c')),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_acc'], 
                   name='Train Acc', line=dict(color='#2ecc71')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_acc'], 
                   name='Val Acc', line=dict(color='#f39c12')),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)
    
    return fig
