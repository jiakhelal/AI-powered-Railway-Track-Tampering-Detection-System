import streamlit as st
from PIL import Image
import numpy as np

from utils.model_utils import load_model, predict
from utils.feature_map_utils import generate_fault_boxes

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Railway Track Fault Detection",
    layout="centered"
)

st.title("🚆 AI-Powered Railway Track Fault Detection")
st.write(
    "Upload a railway track image to detect defects using **DaViT (safety-first logic)**."
)

# ------------------------------------------------------
# LOAD MODEL (NO CACHING — VERY IMPORTANT)
# ------------------------------------------------------
try:
    model = load_model()
except Exception as e:
    st.error("❌ Model failed to load")
    st.exception(e)
    st.stop()

# ------------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Railway Track Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------
    (
        pred,
        confidence,
        tensor,
        non_def_prob,
        def_prob,
        decision_reason
    ) = predict(model, image)

    label = "Defective" if pred == 1 else "Non-Defective"
    color = "red" if pred == 1 else "green"

    st.markdown(f"### 🔍 Prediction: **:{color}[{label}]**")
    st.write(f"**Confidence:** `{confidence:.2f}`")
    st.write(f"**Decision Reason:** {decision_reason}")

    # --------------------------------------------------
    # SIMPLE VISUAL FAULT BOXES (NO OPENCV)
    # --------------------------------------------------
    boxes = generate_fault_boxes(np.array(image).shape, def_prob)

    if boxes:
        st.subheader("⚠️ Highlighted Inspection Regions")
        for box in boxes:
            st.write(
                f"- **{box['label']}** → Region `{box['box']}`"
            )
