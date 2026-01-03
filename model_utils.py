import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
import timm
import torchvision.transforms as T

DEVICE = "cpu"

CLASS_NAMES = ["Non-Defective", "Defective"]

# ======================================================
# Load Model
# ======================================================
def load_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "davit_fastener.pth")

    model = timm.create_model(
        "davit_tiny.msft_in1k",
        pretrained=False,
        num_classes=2
    )

    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


# ======================================================
# Image Transform
# ======================================================
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ======================================================
# HACKATHON ROUND-1 PREDICT LOGIC
# ======================================================
def predict(model, image_pil):
    """
    Round-1 Hackathon Logic:
    Bias toward detecting defects (recall > precision)
    """

    tensor = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    non_def_prob = probs[0, 0].item()
    def_prob = probs[0, 1].item()

    # -------------------------------
    # SAFETY-FIRST DECISION
    # -------------------------------
    if def_prob >= 0.35:
        pred = 1  # Defective
        confidence = def_prob
        decision_reason = "Defect probability above safety threshold"

    elif def_prob >= 0.25:
        pred = 1  # Suspected defect (still Defective for round-1)
        confidence = def_prob
        decision_reason = "Uncertain region – flagged for inspection"

    else:
        pred = 0  # Non-Defective
        confidence = non_def_prob
        decision_reason = "Low defect probability"

    return (
        pred,
        confidence,
        tensor,
        non_def_prob,
        def_prob,
        decision_reason
    )
