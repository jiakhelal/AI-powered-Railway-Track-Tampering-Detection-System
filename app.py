import streamlit as st
from PIL import Image, ImageDraw

from utils.model_utils import load_model, predict
from utils.feature_map_utils import compute_severity, generate_boxes

st.set_page_config(page_title="AI Railway Track Fault Detection", layout="wide")

st.title("🚆 AI-Powered Railway Track Fault Detection")
st.write("Upload a railway track image to detect defects using DaViT with safety-first logic.")

@st.cache_resource
def load_cached_model():
    return load_model()

model = load_cached_model()

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    pred, confidence, reason = predict(model, image)
    severity_score, severity_level = compute_severity(confidence)

    st.subheader("Prediction Result")
    st.write(f"**Class:** {'Defective' if pred else 'Non-Defective'}")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.write(f"**Severity:** {severity_level}")
    st.write(f"**Reason:** {reason}")

    draw = ImageDraw.Draw(image)
    boxes = generate_boxes(image.size[::-1], severity_level)

    for label, x1, y1, x2, y2 in boxes:
        draw.rectangle(
            [x1*image.width, y1*image.height, x2*image.width, y2*image.height],
            outline="red",
            width=4
        )
        draw.text((x1*image.width + 10, y1*image.height + 10), label, fill="red")

    if boxes:
        st.subheader("Fault Localization")
        st.image(image, use_container_width=True)
