import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import pandas as pd

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from utils.inference import EWasteClassifier

st.set_page_config(page_title="Batch Upload", page_icon="📊", layout="wide")
st.title("📊 Batch Image Classification")

@st.cache_resource
def load_classifier():
    return EWasteClassifier(str(Path(__file__).parent.parent / "models" / "resnet50_best.pth"))

try:
    classifier = load_classifier()
    st.success("✅ Model loaded")
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} images uploaded")
    
    if st.button("🚀 Process All", type="primary", width="stretch"):
        progress_bar = st.progress(0)
        results = []
        
        for i, file in enumerate(uploaded_files):
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            try:
                image = Image.open(file)
                result = classifier.predict(image)
                results.append({
                    "Filename": file.name,
                    "Class": result["top_class"],
                    "Confidence": f"{result['top_confidence']:.1f}%",
                    "Time": f"{result['inference_time_ms']:.1f}ms"
                })
            except:
                results.append({
                    "Filename": file.name,
                    "Class": "ERROR",
                    "Confidence": "0%",
                    "Time": "0ms"
                })
        
        progress_bar.empty()
        st.session_state["results"] = results
        st.session_state["files"] = uploaded_files
        st.success(f"✅ Done")

if "results" in st.session_state:
    results = st.session_state["results"]
    df = pd.DataFrame(results)
    
    st.markdown("---")
    st.dataframe(df, width="stretch")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", len(results))
    with col2:
        ok = len([r for r in results if r["Class"] != "ERROR"])
        st.metric("Success", ok)
    
    st.markdown("---")
    csv = df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "results.csv", width="stretch")
else:
    st.info("Upload images to classify them in batch!")
