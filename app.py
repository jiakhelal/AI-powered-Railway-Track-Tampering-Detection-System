import streamlit as st
from PIL import Image, ImageDraw

from utils.model_utils import load_model, predict

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI-Powered Railway Track Fault Detection",
    layout="wide"
)

st.title("🚆 AI-Powered Railway Track Fault Detection")
st.write(
    "Upload a railway track image to detect defects using **DaViT** "
    "(safety-first decision logic)."
)

# ======================================================
# LOAD MODEL (CACHED)
# ======================================================
@st.cache_resource
def load_cached_model():
    return load_model()

model = load_cached_model()

# ======================================================
# SEVERITY COMPUTATION (MINIMAL, SAFE)
# ======================================================
def compute_severity(def_prob):
    score = def_prob

    if score < 0.25:
        level = "Low"
    elif score < 0.4:
        level = "Medium"
    else:
        level = "High"

    return round(score, 2), level

# ======================================================
# DRAW RED BOX (STREAMLIT SAFE)
# ======================================================
def draw_primary_box(image, box):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=6)
    return img

# ======================================================
# IMAGE UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "Upload Railway Track Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ==================================================
    # PREDICTION
    # ==================================================
    pred, confidence, tensor, non_def_prob, def_prob, reason = predict(
        model, image
    )

    label = "Defective" if pred == 1 else "Non-Defective"
    color = "red" if pred == 1 else "green"

    st.markdown(f"### 🔍 Prediction: **:{color}[{label}]**")
    st.markdown(f"**Confidence:** `{round(confidence, 2)}`")
    st.markdown(f"**Decision Reason:** {reason}")

    # ==================================================
    # SEVERITY DISPLAY
    # ==================================================
    severity_score, severity_level = compute_severity(def_prob)

    st.markdown("### ⚠️ Severity Assessment")
    st.markdown(f"- **Severity Score:** `{severity_score}`")
    st.markdown(f"- **Severity Level:** **{severity_level}**")

    # ==================================================
    # VISUAL INSPECTION REGION (ONLY IF DEFECT)
    # ==================================================
    if pred == 1:
        st.markdown("### 🟥 Highlighted Inspection Region")

        # SAME STATIC REGION YOU ARE ALREADY USING
        primary_box = [800, 1200, 3200, 1800]

        boxed_image = draw_primary_box(image, primary_box)

        st.image(
            boxed_image,
            caption="Primary Fault Inspection Region",
            use_container_width=True
        )

        st.markdown(
            f"- **Primary Fault → Region:** `{primary_box}`"
        )
