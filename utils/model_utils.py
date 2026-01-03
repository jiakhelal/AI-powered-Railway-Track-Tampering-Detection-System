import os
import torch
import timm
import torchvision.transforms as T
import urllib.request

# ======================================================
# CONFIG
# ======================================================
DEVICE = "cpu"
CLASS_NAMES = ["Non-Defective", "Defective"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "davit_fastener_state_dict.pth")

MODEL_URL = "https://drive.google.com/uc?id=1Ptp3XbIihWSHh_w2v1cMH2Jb5j50HsHS"

# ======================================================
# DOWNLOAD MODEL
# ======================================================
def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading DaViT model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Model downloaded")

# ======================================================
# SAFE MODEL LOADER (DEPLOYMENT READY)
# ======================================================
def load_model():
    download_model()

    model = timm.create_model(
        "davit_tiny.msft_in1k",
        pretrained=False,
        num_classes=2
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    # 🔥 HANDLE ALL SAVE FORMATS
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise RuntimeError("Unsupported checkpoint format")

    model.load_state_dict(state_dict, strict=False)
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
# PREDICTION LOGIC (SAFETY-FIRST)
# ======================================================
def predict(model, image_pil):
    tensor = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    non_def_prob = probs[0, 0].item()
    def_prob = probs[0, 1].item()

    if def_prob >= 0.35:
        return 1, def_prob, tensor, non_def_prob, def_prob, "High defect probability"
    elif def_prob >= 0.25:
        return 1, def_prob, tensor, non_def_prob, def_prob, "Suspicious region"
    else:
        return 0, non_def_prob, tensor, non_def_prob, def_prob, "Safe region"
