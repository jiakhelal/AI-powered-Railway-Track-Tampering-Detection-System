import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import gdown

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= GOOGLE DRIVE CONFIG =================
DRIVE_FILE_ID = "1Ptp3XbIihWSHh_w2v1cMH2Jb5j50HsHS"
MODEL_PATH = os.path.join("models", "davit_fastener.pth")

# ================= SAFE MODEL DOWNLOAD =================
def download_model():
    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        try:
            torch.load(MODEL_PATH, map_location="cpu")
            print("✅ Model already present. Skipping download.")
            return
        except Exception:
            print("⚠️ Existing model corrupted. Re-downloading...")

    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    print("⬇️ Downloading DaViT model from Google Drive...")

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False,
        fuzzy=True,
        resume=True
    )

    try:
        torch.load(MODEL_PATH, map_location="cpu")
        print("✅ Model download verified.")
    except Exception as e:
        raise RuntimeError("Downloaded model is corrupted.") from e

# ================= LOAD MODEL =================
def load_model(num_classes=2):
    download_model()

    model = timm.create_model(
        "davit_small",
        pretrained=False,
        num_classes=num_classes
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    return model

# ================= IMAGE PREPROCESS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= INFERENCE =================
def run_inference(model, image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    confidence, predicted_class = torch.max(probs, 1)
    return predicted_class.item(), float(confidence.item() * 100)
