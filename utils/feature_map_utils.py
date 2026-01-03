import numpy as np

def compute_severity(confidence):
    if confidence < 0.25:
        return confidence, "Low"
    elif confidence < 0.45:
        return confidence, "Medium"
    else:
        return confidence, "High"


def generate_boxes(image_shape, severity_level):
    h, w = image_shape[:2]

    if severity_level == "High":
        return [("Primary Fault", 0.2, 0.4, 0.8, 0.6)]
    elif severity_level == "Medium":
        return [("Secondary Fault", 0.3, 0.45, 0.7, 0.55)]
    else:
        return []
