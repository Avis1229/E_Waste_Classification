import streamlit as st
import sys
from pathlib import Path
import torch

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

st.set_page_config(page_title="Model Insights", page_icon="📈", layout="wide")
st.title("📈 Model Performance Insights")

model_path = parent_dir / "models" / "resnet50_best.pth"

if model_path.exists():
    checkpoint = torch.load(model_path, map_location="cpu")
    
    st.subheader("🏗️ Model Info")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Architecture", "ResNet50")
    with col2:
        if "epoch" in checkpoint:
            st.metric("Epochs", checkpoint["epoch"])
    with col3:
        if "best_val_acc" in checkpoint:
            st.metric("Val Accuracy", f"{checkpoint['best_val_acc']:.1f}%")
    
    st.markdown("---")
    eval_dir = parent_dir / "evaluation_results"
    if eval_dir.exists():
        cm_path = eval_dir / "confusion_matrix.png"
        if cm_path.exists():
            st.subheader("Confusion Matrix")
            st.image(str(cm_path), width="stretch")
        
        report_path = eval_dir / "classification_report.txt"
        if report_path.exists():
            st.subheader("Classification Report")
            with open(report_path) as f:
                st.text(f.read())
    else:
        st.warning("No evaluation results found")
    
    st.markdown("---")
    st.subheader("Categories")
    cats = ["Mobile", "Laptop", "Keyboards", "Mouses", "TV", "Camera", "Microwave", "Smartwatch"]
    cols = st.columns(4)
    for i, cat in enumerate(cats):
        with cols[i % 4]:
            st.info(cat)
else:
    st.error("Model not found!")
