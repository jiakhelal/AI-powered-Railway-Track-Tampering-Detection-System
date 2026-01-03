import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

from utils.model_utils import load_model, predict
from utils.feature_map_utils import generate_fault_boxes

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI-Powered Railway Track Fault Detection",
    layout="wide"
)

st.title("🤖 AI-Powered Railway Track Fault Detection")
st.write(
    "Upload a railway track image to detect defects using **DaViT** "
    "(safety-first decision logic)."
)

# ======================================================
# LOAD MODEL (CACHED)
# ======================================================
@st.cache_resource(show_spinner="Loading DaViT model...")
def load_cached_model():
    return load_model()

model = load_cached_model()

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "Upload Railway Track Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # ==================================================
    # DISPLAY SMALLER IMAGE (VISUAL ONLY)
    # ==================================================
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(
            image,
            caption="Uploaded Image",
            width=700,
            use_container_width=False
        )

    # ==================================================
    # PREDICTION
    # ==================================================
    pred, confidence, tensor, non_def_prob, def_prob, reason = predict(
        model, image
    )

    label = "Defective" if pred == 1 else "Non-Defective"
    color = "red" if pred == 1 else "green"

    st.markdown(
        f"### 🔍 Prediction: "
        f"<span style='color:{color}'>{label}</span>",
        unsafe_allow_html=True
    )

    st.write(f"**Confidence:** `{confidence:.2f}`")
    st.write(f"**Decision Reason:** {reason}")

    # ==================================================
    # SEVERITY ASSESSMENT
    # ==================================================
    severity_score = confidence

    if severity_score < 0.15:
        severity_level = "Low"
    elif severity_score < 0.35:
        severity_level = "Medium"
    else:
        severity_level = "High"

    st.markdown("## ⚠️ Severity Assessment")
    st.write(f"- **Severity Score:** `{severity_score:.2f}`")
    st.write(f"- **Severity Level:** **{severity_level}**")

    # ==================================================
    # FAULT REGION VISUALIZATION (NO cv2)
    # ==================================================
    h, w = image.size[1], image.size[0]
    boxes = generate_fault_boxes((h, w, 3), severity_score)

    if boxes:
        boxed_img = image.copy()
        draw = ImageDraw.Draw(boxed_img)

        for box in boxes:
            x1, y1, x2, y2 = box["box"]

            # 🔴 Red rectangle
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline="red",
                width=4
            )

            # Label
            draw.text(
                (x1 + 5, max(0, y1 - 20)),
                box["label"],
                fill="red"
            )

        st.markdown("## 🟥 Highlighted Inspection Region")

        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(
                boxed_img,
                width=700,
                use_container_width=False
            )
