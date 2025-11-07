"""
E-Waste Classification App
Main Streamlit application for e-waste classification
"""
import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="E-Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2ecc71;
    }
    .stat-label {
        font-size: 1rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# Main page
st.markdown('<h1 class="main-header">♻️ E-Waste Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Electronic Waste Classification System</p>', unsafe_allow_html=True)

st.markdown("---")

# What is this?
st.markdown("## 🎯 What is E-Waste Classification?")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### The Problem
    - 📈 **50+ million tons** of e-waste generated globally each year
    - ⚠️ Only **20%** is properly recycled
    - 🏭 E-waste contains **toxic materials** (lead, mercury, cadmium)
    - 💰 Also contains **valuable materials** (gold, silver, copper)
    """)

with col2:
    st.markdown("""
    ### Our Solution
    - 🤖 **AI-powered** automated classification
    - ⚡ **Instant** identification of e-waste type
    - ♻️ **Recycling guidance** for each item
    - 🎯 **100% accuracy** on validation set
    """)

st.markdown("---")

# Model Statistics
st.markdown("## 📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">100%</div>
        <div class="stat-label">Validation Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">8</div>
        <div class="stat-label">Categories</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">&lt;50ms</div>
        <div class="stat-label">Inference Time</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">ResNet50</div>
        <div class="stat-label">Model Architecture</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Supported Categories
st.markdown("## 📱 Supported E-Waste Categories")

categories = {
    "⌨️ Keyboards": "Computer keyboards and keypads",
    "📱 Mobile Phones": "Smartphones and feature phones",
    "🖱️ Computer Mice": "Wired and wireless mice",
    "📺 TVs": "All types of televisions",
    "📷 Cameras": "Digital cameras and camcorders",
    "💻 Laptops": "Notebooks and portable computers",
    "🍳 Microwaves": "Microwave ovens",
    "⌚ Smartwatches": "Wearable smart devices"
}

col1, col2, col3, col4 = st.columns(4)
items = list(categories.items())

for i, col in enumerate([col1, col2, col3, col4]):
    with col:
        for j in range(i, len(items), 4):
            cat, desc = items[j]
            st.markdown(f"**{cat}**")
            st.caption(desc)
            st.markdown("")

st.markdown("---")

# How to use
st.markdown("## 🚀 How to Use")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1️⃣ Upload Image
    - Go to **📸 Predict** page
    - Upload an image of e-waste
    - Or use your camera
    """)

with col2:
    st.markdown("""
    ### 2️⃣ Get Classification
    - AI analyzes the image
    - Identifies the category
    - Shows confidence score
    """)

with col3:
    st.markdown("""
    ### 3️⃣ Recycle Properly
    - View recycling tips
    - Learn about hazards
    - Find disposal locations
    """)

st.markdown("---")

# Features
st.markdown("## ✨ Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Core Features
    - ✅ Single image classification
    - ✅ Batch processing (multiple images)
    - ✅ Camera input support
    - ✅ Top-3 predictions with confidence
    - ✅ Detailed recycling guidelines
    - ✅ Export results as CSV
    """)

with col2:
    st.markdown("""
    ### Advanced Features
    - ✅ Model performance insights
    - ✅ Confusion matrix visualization
    - ✅ Per-class accuracy analysis
    - ✅ Real-time inference
    - ✅ Responsive design
    - ✅ User-friendly interface
    """)

st.markdown("---")

# Quick start
st.info("👉 **Get Started:** Click on **📸 Predict** in the sidebar to classify your first e-waste item!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using PyTorch and Streamlit</p>
    <p>♻️ Help save the planet by recycling e-waste properly!</p>
</div>
""", unsafe_allow_html=True)
