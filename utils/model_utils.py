import os
import torch
import timm
import torchvision.transforms as T
import urllib.request

# ===============================
# CONFIG
# ===============================
DEVICE = "cpu"
CLASS_NAMES = ["Non-Defective", "Defective"]

MODEL_URL = "https://drive.google.com/uc?id=1n1aAuUGxuNUEorj_zT3koe03ZvC4D-R7"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "davit_fastener.pth")

# ===============================
# DOWNLOAD MODEL
# ===============================
def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ===============================
# LOAD MODEL (🔥 FIXED)
# ===============================
def load_model():
    download_model()

    model = timm.create_model(
        "davit_tiny.msft_in1k",
        pretrained=False,
        num_classes=2
    )

    # 🔥 THIS LINE FIXES YOUR ERROR
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# ===============================
# TRANSFORM
# ===============================
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ===============================
# PREDICT
# ===============================
def predict(model, image_pil):
    tensor = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    non_def_prob = probs[0, 0].item()
    def_prob = probs[0, 1].item()

    if def_prob >= 0.35:
        pred = 1
        confidence = def_prob
        decision_reason = "Defect probability above safety threshold"
    elif def_prob >= 0.25:
        pred = 1
        confidence = def_prob
        decision_reason = "Uncertain region – flagged for inspection"
    else:
        pred = 0
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
