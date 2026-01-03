import numpy as np

# ======================================================
# SEVERITY COMPUTATION (TOKEN-BASED)
# ======================================================
def compute_severity(token_energy):
    """
    token_energy: 1D numpy array from model tokens
    """
    high_tokens = token_energy > 0.7
    score = float(np.sum(high_tokens) / len(token_energy))

    if score < 0.03:
        level = "Low"
    elif score < 0.1:
        level = "Medium"
    else:
        level = "High"

    return score, level


# ======================================================
# FAULT BOX GENERATION (NO CV2)
# ======================================================
def generate_fault_boxes(image_shape, severity_score):
    """
    Simple heuristic inspection regions
    Streamlit-safe (no cv2)
    """

    h, w = image_shape[:2]
    boxes = []

    if severity_score > 0.3:
        boxes.append({
            "label": "Primary Fault",
            "color": "red",
            "box": [int(0.2*w), int(0.4*h), int(0.8*w), int(0.6*h)]
        })

    elif severity_score > 0.15:
        boxes.append({
            "label": "Secondary Fault",
            "color": "blue",
            "box": [int(0.3*w), int(0.45*h), int(0.7*w), int(0.55*h)]
        })

    return boxes
