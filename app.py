import streamlit as st
from utils.model_utils import load_model, predict
from utils.feature_map_utils import generate_fault_boxes
from PIL import Image

st.set_page_config(page_title="AI Railway Track Fault Detection", layout="wide")

st.title("🚆 AI-Powered Railway Track Fault Detection")

@st.cache_data(show_spinner=False)
def get_model():
    return load_model()

model = get_model()
