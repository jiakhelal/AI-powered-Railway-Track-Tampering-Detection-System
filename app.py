import streamlit as st
from PIL import Image

from utils.model_utils import load_model, predict, CLASS_NAMES
from utils.feature_map_utils import generate_fault_boxes

st.set_page_config(page_title="AI Railway Track Fault Detection", layout="centered")

@st.cache_resource
def load_cached_model():
    return load_model()

st.title("🚆 AI-Powered Railway Track Fault Detection")
st.write("Upload a railway track image to detect defects using DaViT (safety-first logic).")

model = load_cached_model()

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    pred, confidence, reason = predict(model, image)

    st.subheader("Prediction")
    st.write(f"**Class:** {CLASS_NAMES[pred]}")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.write(f"**Decision:** {reason}")

    severity_score = confidence
    boxes = generate_fault_boxes(image.size[::-1], severity_score)

    if boxes:
        st.subheader("⚠️ Detected Regions")
        for b in boxes:
            st.write(f"- {b['label']} region detected")
