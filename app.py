import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch

# ===============================
# IMPORT YOUR UTILS (CORRECT)
# ===============================
from utils.model_utils import load_model, predict
from utils.feature_map_utils import generate_fault_boxes

# ===============================
# STREAMLIT PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI-Powered Railway Track Fault Detection",
    layout="centered"
)

st.title("🚆 AI-Powered Railway Track Fault Detection")
st.write(
    "Upload a railway track image to detect **defects / tampering** "
    "using **DaViT Vision Transformer** with **safety-first logic**."
)

# ===============================
# LOAD MODEL (NO CACHING PICKLE)
# ===============================
model = load_model()


# ===============================
# IMAGE UPLOAD
# ===============================
uploaded_file = st.file_uploader(
    "Upload railway track image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.stop()

# ===============================
# IMAGE PREP
# ===============================
image = Image.open(uploaded_file).convert("RGB")
img_np = np.array(image)

st.subheader("Uploaded Image")
st.image(image, use_container_width=True)

# ===============================
# MODEL PREDICTION
# ===============================
(
    pred,
    confidence,
    tensor,
    non_def_prob,
    def_prob,
    decision_reason
) = predict(model, image)

final_class = "Defective" if pred == 1 else "Non-Defective"

# ===============================
# RESULTS
# ===============================
st.subheader("Prediction Result")

st.markdown(f"""
**Final Class:** `{final_class}`  
**Confidence:** `{confidence:.2f}`  

**Non-Defective Probability:** `{non_def_prob:.2f}`  
**Defective Probability:** `{def_prob:.2f}`  

**Decision Reason:** {decision_reason}
""")

if confidence < 0.5:
    st.warning("⚠️ Uncertain prediction – flagged for manual inspection")

# ===============================
# FAULT BOX GENERATION
# ===============================
boxes = generate_fault_boxes(
    image_shape=img_np.shape,
    severity_score=confidence
)

# ===============================
# DRAW BOXES USING PIL (NO CV2)
# ===============================
draw_img = image.copy()
draw = ImageDraw.Draw(draw_img)

for b in boxes:
    x1, y1, x2, y2 = b["box"]
    color = "red" if b["color"] == "red" else "blue"

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=color,
        width=4
    )

    draw.text(
        (x1 + 5, y1 + 5),
        b["label"],
        fill=color
    )

# ===============================
# DISPLAY EXPLANATION
# ===============================
st.subheader("Model Explanation")

if len(boxes) == 0:
    st.info("No dominant fault regions detected.")
else:
    st.image(draw_img, use_container_width=True)
    st.caption("Primary (RED) / Secondary (BLUE) fault regions")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption(
    "⚙️ Deployed using Streamlit Community Cloud | "
    "DaViT Vision Transformer | Safety-First AI"
)



