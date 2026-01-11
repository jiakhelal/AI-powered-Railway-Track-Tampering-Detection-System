import streamlit as st
from PIL import Image, ImageDraw
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

# IMPORTANT: utils/inference.py must exist
from utils.inference import load_model, run_inference


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Railway AI Command Center",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= NEON UI CSS =================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0b1220, #05070c);
    color: #e8ecf1;
    font-family: 'Segoe UI', system-ui;
}
.neon-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg, #00e5ff, #3fa9f5, #7c7cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    color: #9aa4af;
    margin-bottom: 40px;
}
.glass {
    background: rgba(18,24,38,0.75);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 24px;
    border: 1px solid rgba(63,169,245,0.25);
    box-shadow: 0 0 25px rgba(63,169,245,0.15);
}
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
}
.high { background:#ff3c3c; box-shadow:0 0 20px #ff3c3c; }
.medium { background:#ffaa00; box-shadow:0 0 20px #ffaa00; }
.low { background:#00ffaa; box-shadow:0 0 20px #00ffaa; }
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ================= HEADER =================
st.markdown("<div class='neon-title'>Railway AI Command Center</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Autonomous Inspection · Tampering Intelligence · Human Authorization</div>",
    unsafe_allow_html=True
)

# ================= METADATA =================
st.markdown(f"""
<div class="glass">
<b>Inspection ID:</b> IR-2026-0142 &nbsp;&nbsp;
<b>Section:</b> NDLS–CNB &nbsp;&nbsp;
<b>Mode:</b> Drone Surveillance<br>
<b>Timestamp:</b> {datetime.now().strftime("%d %b %Y | %H:%M IST")}
</div>
""", unsafe_allow_html=True)

# ================= IMAGE UPLOAD =================
st.markdown("<div class='glass'><h3>📷 Drone Capture</h3>", unsafe_allow_html=True)
img = st.file_uploader("Upload track image", ["jpg", "png", "jpeg"])
st.markdown("</div>", unsafe_allow_html=True)

# ================= MAIN PIPELINE =================
if img:
    image = Image.open(img).convert("RGB")
    st.image(image, width=520)

    # -------- MODEL INFERENCE --------
    predicted_class, confidence = run_inference(model, image)

    # -------- DECISION LOGIC (CORRECT & SAFE) --------
    if predicted_class == 0:
        # Model predicts NORMAL
        if confidence >= 70:
            prediction = "Normal Track"
            status = "NORMAL CONDITION"
            severity = "LOW"
            sev_class = "low"
            color = "#00ffaa"
            explanation = (
                "Track visually matches learned normal patterns. "
                "No abnormal structural elements detected."
            )
        else:
            prediction = "Normal Track"
            status = "UNCERTAIN – REQUIRES REVIEW"
            severity = "LOW"
            sev_class = "low"
            color = "#00ffaa"
            explanation = (
                "Low-confidence normal prediction. "
                "Image quality or viewing angle may affect certainty."
            )

    else:
        # Model predicts DEFECTIVE
        prediction = "Defective Track"

        if confidence >= 75:
            status = "SUSPECTED INTENTIONAL TAMPERING"
            severity = "HIGH"
            sev_class = "high"
            color = "#ff3c3c"
            explanation = (
                "High-confidence abnormal object or structural inconsistency detected "
                "within the rail corridor. Pattern is inconsistent with natural wear "
                "and may indicate external interference."
            )

        elif confidence >= 55:
            status = "SUSPECTED ANOMALY (CAUSE UNKNOWN)"
            severity = "MEDIUM"
            sev_class = "medium"
            color = "#ffaa00"
            explanation = (
                "Moderate-confidence abnormality detected. "
                "Could be debris, ballast displacement, or early-stage defect. "
                "Human verification required."
            )

        else:
            status = "LOW-CONFIDENCE ANOMALY"
            severity = "LOW"
            sev_class = "low"
            color = "#00ffaa"
            explanation = (
                "Weak anomaly signal detected. "
                "Likely visual noise or benign variation."
            )

    # -------- ALERT PANEL --------
    st.markdown(f"""
    <div class="glass">
    <h3>🚨 AI Alert</h3>
    <b>Status:</b> {status}<br>
    <b>Prediction:</b> {prediction}<br>
    <b>Confidence:</b> {confidence:.1f}%<br>
    <span class="badge {sev_class}">{severity} RISK</span>
    </div>
    """, unsafe_allow_html=True)

    # -------- AI EXPLANATION --------
    st.markdown(f"""
    <div class="glass">
    <h3>🧠 AI Reasoning</h3>
    {explanation}
    </div>
    """, unsafe_allow_html=True)

    # -------- VISUAL OVERLAY --------
    draw = ImageDraw.Draw(image)
    w, h = image.size

    if predicted_class == 1:
        x1, y1, x2, y2 = int(w * 0.3), int(h * 0.45), int(w * 0.7), int(h * 0.7)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        draw.rectangle([x1, y1 - 32, x2, y1], fill=color)
        draw.text((x1 + 8, y1 - 26), f"{status} | {confidence:.1f}%", fill="white")

    st.image(image, width=520)

    # -------- HUMAN AUTHORIZATION --------
    st.markdown("<div class='glass'><h3>👤 Human Control Panel</h3>", unsafe_allow_html=True)

    action = st.selectbox(
        "Authorized Railway Action",
        [
            "Monitor Only",
            "Impose Speed Restriction",
            "Schedule Maintenance",
            "Immediate Track Possession",
            "Suspend Traffic"
        ]
    )

    remarks = st.text_area("Officer Remarks")

    # -------- PDF REPORT --------
    def generate_pdf():
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        c.setFont("Helvetica", 11)
        y = 800

        for line in [
            "Railway AI Inspection Report",
            f"Status: {status}",
            f"Prediction: {prediction}",
            f"Confidence: {confidence:.1f}%",
            f"Authorized Action: {action}",
            f"Remarks: {remarks}",
            f"Timestamp: {datetime.now().strftime('%d %b %Y | %H:%M IST')}",
        ]:
            c.drawString(50, y, line)
            y -= 20

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer

    if st.button("✅ Authorize & Generate Report"):
        pdf = generate_pdf()
        st.success("Decision Authorized")
        st.download_button(
            "📄 Download Inspection Report",
            data=pdf,
            file_name="railway_ai_report.pdf",
            mime="application/pdf"
        )
