import streamlit as st
import numpy as np
from PIL import Image

from utils.model_utils import load_model, predict, CLASS_NAMES
from utils.feature_map_utils import (
    compute_severity,
    generate_fault_boxes
)


# ======================================================
# Streamlit Page Config
# ======================================================
st.set_page_config(
    page_title="Railway Track Fault Detection",
    layout="centered"
)

st.markdown(
    "<h2 style='text-align:center;'>🚆 Railway Track Fault Detection</h2>",
    unsafe_allow_html=True
)

# ======================================================
# Load Model (Cached)
# ======================================================
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ======================================================
# Upload Image
# ======================================================
uploaded = st.file_uploader(
    "Upload Railway Track Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)

    st.image(
        image,
        caption="Uploaded Image",
        width=400
    )

    st.markdown("---")

    # ==================================================
    # Detect Button
    # ==================================================
    if st.button("🔍 Detect Fault"):
        with st.spinner("Analyzing railway track..."):

            # ------------------------------
            # Prediction (Round-1 Logic)
            # ------------------------------
            (
                pred,
                confidence,
                tensor,
                non_def_prob,
                def_prob,
                decision_reason
            ) = predict(model, image)

            st.subheader("Prediction Result")
            st.write(f"**Final Class:** {CLASS_NAMES[pred]}")
            st.write(f"**Confidence:** {confidence:.2f}")
            st.write(f"**Decision Reason:** {decision_reason}")
            st.write(f"Non-Defective Probability: `{non_def_prob:.2f}`")
            st.write(f"Defective Probability: `{def_prob:.2f}`")

            if confidence < 0.65:
                st.warning("⚠️ Uncertain prediction – flagged for inspection")

            # ------------------------------
            # Token Importance
            # ------------------------------
            token_energy = generate_feature_map(
                model,
                tensor
            )

            # ------------------------------
            # Severity
            # ------------------------------
            severity_score, severity_level = compute_severity(
                token_energy
            )

            # ------------------------------
            # Draw Boxes (FORCED if Defective)
            # ------------------------------
            boxed_img = draw_boxes(
                img_np,
                token_energy,
                severity_score,
                force=(pred == 1)  # 🔴 FORCE boxes for Defective
            )

            # ------------------------------
            # Explanation Output
            # ------------------------------
            st.subheader("Model Explanation")
            st.write(f"**Severity Score:** {severity_score:.2f}")
            st.write(f"**Severity Level:** {severity_level}")

            st.image(
                boxed_img,
                caption="Primary (RED) / Secondary (BLUE) Fault Regions",
                width=500
            )


