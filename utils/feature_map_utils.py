import torch
import numpy as np
import cv2

# =========================================================
# TOKEN IMPORTANCE (USED INTERNALLY)
# =========================================================
def generate_feature_map(model, tensor):
    """
    Returns token-level importance (1D).
    Kept name for backward compatibility with app.py
    """

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        feats = model.forward_features(tensor)  # [1, N, C]

    # Remove CLS token if present
    if feats.shape[1] > 100:
        feats = feats[:, 1:, :]

    energy = torch.norm(feats, dim=2).squeeze(0)
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)

    return energy.cpu().numpy()


# =========================================================
# SEVERITY COMPUTATION
# =========================================================
def compute_severity(token_energy):
    high_tokens = token_energy > 0.7
    score = float(np.sum(high_tokens) / len(token_energy))

    if score < 0.03:
        level = "Low"
    elif score < 0.1:
        level = "Medium"
    else:
        level = "High"

    return score, level


# =========================================================
# DRAW BOXES (THICK + LABELED + FORCED)
# =========================================================
def draw_boxes(img, token_energy, severity_score, force=False):

    out = img.copy()
    h, w = img.shape[:2]
    n = len(token_energy)

    if severity_score < 0.03 and not force:
        return out

    idx_sorted = np.argsort(token_energy)[::-1]

    primary_tokens = idx_sorted[: max(1, int(0.05 * n))]
    secondary_tokens = idx_sorted[int(0.05 * n): max(2, int(0.15 * n))]

    def draw_labeled_box(y1, y2, color, label):
        thickness = 4
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8

        cv2.rectangle(out, (0, y1), (w, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)
        cv2.rectangle(
            out,
            (5, y1 + 5),
            (5 + tw + 10, y1 + th + 20),
            color,
            -1
        )
        cv2.putText(
            out,
            label,
            (10, y1 + th + 15),
            font,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    def tokens_to_box(tokens, color, label):
        if len(tokens) == 0:
            return False

        ys = (tokens / n) * h
        y1, y2 = int(np.min(ys)), int(np.max(ys))

        if y2 - y1 < 60:
            y1 = int(0.35 * h)
            y2 = int(0.65 * h)

        draw_labeled_box(y1, y2, color, label)
        return True

    drawn = tokens_to_box(primary_tokens, (0, 0, 255), "Primary Fault")
    tokens_to_box(secondary_tokens, (255, 0, 0), "Secondary Fault")

    if not drawn and force:
        y1, y2 = int(0.35 * h), int(0.65 * h)
        draw_labeled_box(y1, y2, (0, 0, 255), "Inspection Region")

    return out
