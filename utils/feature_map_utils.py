import numpy as np

def generate_fault_boxes(image_shape, severity_score):
    """
    Simple heuristic boxes for Streamlit Cloud
    (NO OpenCV, NO torch ops)
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
