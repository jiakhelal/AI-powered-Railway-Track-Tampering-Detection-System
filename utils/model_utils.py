import os
import torch
import timm
import torchvision.transforms as T
import urllib.request

DEVICE = "cpu"
CLASS_NAMES = ["Non-Defective", "Defective"]

MODEL_URL = "https://drive.google.com/uc?id=1n1aAuUGxuNUEorj_zT3koe03ZvC4D-R7"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "davit_fastener.pth")

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def load_model():
    download_model()

    model = timm.create_model(
        "davit_tiny.msft_in1k",
        pretrained=False,
        num_classes=2
    )

    # 🔥 CRITICAL FIX
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def predict(model, image_pil):
    tensor = TRANSFORM(image_pil).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)

    non_def = probs[0, 0].item()
    defect = probs[0, 1].item()

    if defect >= 0.35:
        return 1, defect, "High risk defect"
    elif defect >= 0.25:
        return 1, defect, "Suspected defect"
    else:
        return 0, non_def, "No defect detected"
