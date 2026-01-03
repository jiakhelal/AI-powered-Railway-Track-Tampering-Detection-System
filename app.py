import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw

# ===============================
# IMPORT YOUR UTILS (SAFE)
# ===============================
from utils.model_utils import load_model, predict, CLASS_NAMES
from utils.feature_map_utils import compute_severity, generate_fault_boxes


# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="AI Railway Track Fault Detection",
    layout="centered"
)

st.title("🚆 AI-Powered Railway Track Fault Detection")
st.write(
    "Upload a railway track image to detect **tampering / defects** using "
    "**DaViT Vision Transformer** with safety-first decision logic."
)

# ===============================
# LOAD MODEL (CACHED)
# ===============================
@st.cache_resource
def load_cached_model():
    return load_model()

model = load_cached_model()


# ===============================
# IMAGE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload Railway Track Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Fault"):
        with st.spinner("Analyzing image..."):

            # -------------------------------
            # PREDICTION
            # -------------------------------
            (
                pred,
                confidence,
                tensor,
                non_def_prob,
                def_prob,
                decision_reason
            ) = predict(model, image)

            label = CLASS_NAMES[pred]

            # -------------------------------
            # FEATURE TOKENS → SEVERITY
            # -------------------------------
            with torch.no_grad():
                feats = model.forward_features(tensor)
                feats = feats[:, 1:, :]  # remove CLS token
                token_energy = torch.norm(feats, dim=2).squeeze().cpu().numpy()

            severity_score, severity_level = compute_severity(token_energy)

            # -------------------------------
            # DISPLAY RESULTS
            # -------------------------------
            st.subheader("🧠 Prediction Result")
            st.write(f"**Prediction:** `{label}`")
            st.write(f"**Confidence:** `{confidence:.2f}`")
            st.write(f"**Decision Reason:** {decision_reason}")

            st.subheader("⚠️ Severity Analysis")
            st.write(f"**Severity Score:** `{severity_score:.3f}`")
            st.write(f"**Severity Level:** `{severity_level}`")

            # -------------------------------
            # DRAW FAULT BOXES (NO CV2)
            # -------------------------------
            img_np = np.array(image)
            boxes = generate_fault_boxes(img_np.shape, severity_score)

            if boxes:
                draw = ImageDraw.Draw(image)

                for b in boxes:
                    x1, y1, x2, y2 = b["box"]
                    color = b["color"]

                    draw.rectangle(
                        [(x1, y1), (x2, y2)],
                        outline=color,
                        width=5
                    )
                    draw.text(
                        (x1 + 10, y1 + 10),
                        b["label"],
                        fill=color
                    )

                st.subheader("📍 Fault Localization")
                st.image(image, use_container_width=True)
            else:
                st.info("No critical fault region detected.")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "Built using **DaViT Vision Transformer**, Streamlit Cloud & Safety-First AI Logic"
)
