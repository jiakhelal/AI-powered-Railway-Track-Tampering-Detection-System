import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
import timm
import torchvision.transforms as T
import urllib.request

DEVICE = "cpu"

CLASS_NAMES = ["Non-Defective", "Defective"]

# ======================================================
# MODEL DOWNLOAD CONFIG
# ======================================================
MODEL_URL = "https://drive.google.com/uc?id=1n1aAuUGxuNUEorj_zT3koe03ZvC4D-R7"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "davit_fastener.pth")

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading DaViT model from Google Drive...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Model downloaded successfully.")

# ======================================================
# LOAD MODEL
# ======================================================
def load_model():
    download_model()

    model = timm.create_model(
        "davit_tiny.msft_in1k",
        pretrained=False,
        num_classes=2
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)

    model.to(DEVICE)
    model.eval()
    return model

# ======================================================
# IMAGE TRANSFORM
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
        pred = 1  # Suspected defect
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
