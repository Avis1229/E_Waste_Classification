import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import os
import urllib.request

# determine absolute path to model so uploads work correctly from anywhere
BASE_DIR = Path(__file__).parent.parent

# --- यहाँ बदलाव किया गया है: इंटरनेट से मॉडल डाउनलोड करने का लॉजिक ---
MODEL_PATH = os.path.join(os.getcwd(), "resnet50_best.pth")

# अगर फ़ाइल पहले से डाउनलोड नहीं है, तो उसे Hugging Face से डाउनलोड करें
if not os.path.exists(MODEL_PATH):
    with st.spinner("AI Model डाउनलोड हो रहा है, कृपया कुछ सेकंड रुकें..."):
        # यह एक स्टेबल ResNet50 मॉडल का डायरेक्ट लिंक है
        url = "https://huggingface.co"
        urllib.request.urlretrieve(url, MODEL_PATH)
# -----------------------------------------------------------------

print("Model Path:", MODEL_PATH)

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from utils.inference import EWasteClassifier, get_recycling_tips
from utils.visualize import plot_confidence_bar

st.set_page_config(page_title="Predict", page_icon="📸", layout="wide")
st.title("📸 Single Image Classification")

@st.cache_resource
def load_classifier():
    return EWasteClassifier(str(MODEL_PATH))

try:
    classifier = load_classifier()
    st.success("✅ Model loaded")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
with col2:
    camera_image = st.camera_input("Camera")

image_source = camera_image if camera_image else uploaded_file

if image_source:
    image = Image.open(image_source)
    # Streamlit के नए वर्ज़न में use_container_width=True इस्तेमाल होता है
    st.image(image, use_container_width=True)
    
    if st.button("🚀 Classify", type="primary"):
        with st.spinner("Analyzing..."):
            result = classifier.predict(image)
            top_class = result["top_class"]
            top_conf = result["top_confidence"]
            st.success(f"**{top_class}** - {top_conf:.1f}% confidence")
            
            tips = get_recycling_tips(top_class)
            st.markdown("### ♻️ Recycling Information")
            st.info(tips["description"])
            
            st.markdown("**Tips:**")
            for tip in tips["tips"]:
                st.markdown(f"- {tip}")
            
            if tips["hazards"]:
                st.warning(f"⚠️ **Safety Warning:** {tips['hazards']}")
else:
    st.info("Upload an image or use camera to classify e-waste")
