---
title: E-Waste Classifier
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.51.0
app_file: app.py
pinned: false
license: mit
tags:
  - computer-vision
  - image-classification
  - pytorch
  - resnet
  - e-waste
  - recycling
  - environmental
---

# E-Waste Classification Project

Deep Learning model for classifying electronic waste into 8 categories using ResNet50.

## 🎯 Project Overview

This project implements a CNN-based classifier to identify different types of e-waste including:
- **Keyboards** ⌨️
- **Mobile** 📱
- **Mouses** 🖱️
- **TV** 📺
- **Camera** 📷
- **Laptop** 💻
- **Microwave** 🍲
- **Smartwatch** ⌚

**Model Performance:**
- ✅ **100% Validation Accuracy**
- ✅ **ResNet50 Architecture**
- ✅ **~24.7M Parameters**
- ✅ **<50ms Inference Time**

## 🌟 Features

- 🖼️ **Single Image Classification** - Upload or capture images for instant prediction
- 📊 **Batch Processing** - Classify multiple images at once with CSV/Excel export
- 📈 **Model Insights** - View confusion matrix, accuracy metrics, and architecture details
- ♻️ **Recycling Tips** - Get disposal instructions and safety warnings for each e-waste category
- 🎨 **Modern UI** - Clean, responsive Streamlit interface with custom green theme

## 📁 Project Structure

```
ewaste_classifier/
├── data/                   # Dataset directory
│   ├── train/             # Training images (844 samples)
│   ├── val/               # Validation images (167 samples)
│   └── test/              # Test images (172 samples)
├── src/                   # Source code
│   ├── dataset.py         # Data loading & augmentation
│   ├── model.py           # Model architecture (ResNet50)
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation metrics
│   ├── predict.py         # Inference function
│   ├── config.py          # Training configurations
│   ├── split_dataset.py   # Dataset splitting utility
│   └── validate_data.py   # Data validation script
├── models/                # Saved model checkpoints
│   └── resnet50_best.pth  # Trained ResNet50 model
├── utils/                 # Utility functions
│   ├── inference.py       # Streamlit inference wrapper
│   └── visualize.py       # Plotting and visualization
├── pages/                 # Streamlit multi-page app
│   ├── 1_Predict.py       # Single image classification
│   ├── 2_Batch_Upload.py  # Batch processing
│   ├── 3_Model_Insights.py # Model performance metrics
│   └── 4_About.py         # Project information
├── .streamlit/            # Streamlit configuration
│   └── config.toml        # Custom green theme
├── evaluation_results/    # Evaluation outputs
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   └── classification_report.txt
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── plan.txt               # Project roadmap
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended for training)
- 4GB+ VRAM for training

### 1. Installation

```bash
# Clone the repository
cd e_waste

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Start - Run the Web App

The model is already trained! Just run:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### 3. Dataset Setup (if training from scratch)

**Option A: Download from Kaggle**
- Search for "e-waste dataset" or "electronic waste classification" on Kaggle
- Download and extract to the `data/` folder
- Organize into train/val/test splits with class subdirectories

**Option B: Manual Dataset Creation**
1. Create class subdirectories in `data/train/`, `data/val/`, and `data/test/`
2. Collect at least 500 images per class
3. Organize images into respective class folders

**Expected Structure:**
```
data/
├── train/
│   ├── batteries/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── cables/
│   ├── chargers/
│   └── ...
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

### 3. Validate Your Dataset

```bash
cd src
python validate_data.py
```

This will check for:
- ✅ Corrupted images
- ✅ Class imbalance
- ✅ Image quality issues
- ✅ Dataset structure

## 📊 Project Status

### Phase 1: Data ✅ COMPLETED
- [x] Create project structure
- [x] Setup requirements.txt
- [x] Create dataset.py with PyTorch DataLoader
- [x] Create data validation script
- [x] Download and organize dataset (1,183 total images)
- [x] Split dataset: 844 train / 167 val / 172 test

### Phase 2: Model ✅ COMPLETED
- [x] Build baseline CNN model
- [x] Implement transfer learning (ResNet50)
- [x] Setup training pipeline with optimizers and schedulers
- [x] Add early stopping and checkpointing
- [x] Create training configuration system
- [x] Implement evaluation metrics and visualizations
- [x] Create prediction/inference script

### Phase 3: Training & Evaluation ✅ COMPLETED
- [x] Train ResNet50 model on GPU (NVIDIA RTX 2050)
- [x] Two-stage training: frozen backbone (20 epochs) + fine-tuning (30 epochs)
- [x] Achieve 100% validation accuracy
- [x] Generate confusion matrix and classification reports
- [x] Track metrics (accuracy, F1-score, loss)
- [x] Optimize inference speed (<50ms per image)

### Phase 4: Streamlit Web App ✅ COMPLETED
- [x] Build main homepage with project overview
- [x] Create single image prediction interface
- [x] Add batch upload and processing feature
- [x] Build model insights dashboard
- [x] Create About page with documentation
- [x] Implement recycling tips database
- [x] Add custom green theme
- [x] Deploy locally with hot-reload

## 💻 Usage

### 🌐 Run Web Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

**Available Pages:**
- **Home** - Project overview and statistics
- **Predict** - Upload or capture single image for classification
- **Batch Upload** - Process multiple images and export results
- **Model Insights** - View performance metrics and architecture
- **About** - Project documentation and information

### 🏋️ Train Model (from scratch)

```bash
# Two-stage training (recommended)
python train_resnet50_twostage.py

# Custom training
python src/train.py --model resnet50 --epochs 50 --batch_size 64 --lr 1e-4
```

### 📊 Evaluate Model

```bash
python src/evaluate.py --model models/resnet50_best.pth --test_dir data/test
```

Results will be saved to `evaluation_results/` directory.

### 🔮 Single Prediction (CLI)

```bash
python src/predict.py --model models/resnet50_best.pth --image path/to/image.jpg
```

### ✅ Validate Dataset

```bash
python src/validate_data.py
```

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 100.0% |
| **Model Architecture** | ResNet50 |
| **Total Parameters** | ~24.7M |
| **Trainable Parameters** | ~1.2M (final layers) |
| **Inference Time** | <50ms per image |
| **Model Size** | ~95MB |
| **Training Device** | NVIDIA RTX 2050 (4GB VRAM) |
| **Training Time** | ~50 epochs (2 stages) |

### Training Strategy

1. **Stage 1 - Feature Extraction (20 epochs)**
   - Freeze ResNet50 backbone
   - Train only classifier head
   - Learning rate: 0.001
   - Batch size: 64

2. **Stage 2 - Fine-Tuning (30 epochs)**
   - Unfreeze all layers
   - End-to-end training
   - Learning rate: 0.0001
   - Batch size: 64

### Dataset Statistics

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 844 | 71.3% |
| Validation | 167 | 14.1% |
| Test | 172 | 14.5% |
| **Total** | **1,183** | **100%** |

**Classes:** 8 e-waste categories (balanced distribution)

## 📚 Technologies Used

### Deep Learning & ML
- **PyTorch 2.7.1** - Deep learning framework
- **torchvision** - Pre-trained models and transforms
- **Albumentations 1.3+** - Advanced data augmentation
- **CUDA 12.9** - GPU acceleration

### Web Application
- **Streamlit 1.51.0** - Interactive web interface
- **Plotly 6.4.0** - Interactive visualizations
- **Pillow** - Image processing
- **pandas** - Data manipulation
- **openpyxl** - Excel export

### Development Tools
- **NumPy** - Numerical computing
- **scikit-learn** - Metrics and evaluation
- **tqdm** - Progress bars

## 🎨 Web Application Features

### 1. 📸 Single Image Classification
- Upload image or use camera
- Real-time prediction with confidence scores
- Recycling tips and safety warnings
- Downloadable results

### 2. 📊 Batch Processing
- Upload multiple images at once
- Progress tracking
- Results table with CSV/Excel export
- Class distribution visualization
- Image grid with predictions

### 3. 📈 Model Insights
- Confusion matrix visualization
- Per-class accuracy charts
- Classification reports
- Training configuration
- Architecture details
- Performance benchmarks

### 4. ℹ️ About & Documentation
- Project overview
- Technical details
- Dataset information
- Future improvements roadmap

## 🔧 Technical Highlights

- ✅ **Transfer Learning** - Pre-trained ResNet50 on ImageNet
- ✅ **Two-Stage Training** - Frozen → Fine-tuned for better convergence
- ✅ **Data Augmentation** - Rotation, scaling, brightness, noise
- ✅ **Mixed Precision** - FP16 training for faster computation
- ✅ **Batch Normalization** - Stable training
- ✅ **Dropout Regularization** - Prevent overfitting
- ✅ **Early Stopping** - Automatic training termination
- ✅ **Model Checkpointing** - Save best model automatically
- ✅ **GPU Optimization** - Efficient memory usage (4GB VRAM)

## 🚀 Future Improvements

- [ ] Deploy to cloud (Streamlit Cloud / Heroku)
- [ ] Add real-time video classification
- [ ] Mobile app development
- [ ] Multi-label classification (damaged items)
- [ ] Ensemble methods for better accuracy
- [ ] Fine-grained sub-categorization
- [ ] Integration with recycling center APIs
- [ ] Multi-language support
- [ ] Offline mode support
- [ ] Model quantization for edge devices

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **Dataset** - E-waste classification dataset from Kaggle
- **PyTorch** - Deep learning framework
- **Streamlit** - Web application framework
- **ResNet** - Architecture by Microsoft Research

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Status**: ✅ **All Phases Complete** | Ready for Production

**Live Demo**: Run `streamlit run app.py` to see the application in action!
