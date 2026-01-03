import streamlit as st
from PIL import Image
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

    # 🔽 DISPLAY SMALLER IMAGE (NO RESIZE OF ACTUAL IMAGE)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=False,
            width=700   # 👈 controls visual size only
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
    # SEVERITY (USING DEFECT PROBABILITY)
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
    # FAULT BOX (VISUAL)
    # ==================================================
    h, w = image.size[1], image.size[0]
    boxes = generate_fault_boxes((h, w, 3), severity_score)

    if boxes:
        import cv2

        img_np = np.array(image)

        for box in boxes:
            x1, y1, x2, y2 = box["box"]
            cv2.rectangle(
                img_np,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                3
            )
            cv2.putText(
                img_np,
                box["label"],
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2
            )

        boxed_img = Image.fromarray(img_np)

        st.markdown("## 🟥 Highlighted Inspection Region")

        # 🔽 SMALLER DISPLAY AGAIN
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(
                boxed_img,
                use_container_width=False,
                width=700
            )
