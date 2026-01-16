# app.py
# RayScan – Lung X-ray Image Classification Web App

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import streamlit as st
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torch.nn.functional as F
import tensorflow as tf # type: ignore
import pydicom
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# =========================
# App Configuration
# =========================
st.set_page_config(
    page_title="RayScan – Lung X-ray AI",
    page_icon="🩻",
    layout="wide"
)

# =========================
# Branding
# =========================
APP_NAME = "RayScan"
TAGLINE = "AI-powered early insight from chest X-rays"

# =========================
# Load Logo
# =========================
# Get the path to the current file's directory
BASE_DIR = Path(__file__).resolve().parent

# Define the path to the logo
LOGO_PATH = str(BASE_DIR / "assets" / "Logo.jpg")

print(LOGO_PATH)

# =========================
#Rebuild the model architecture
# =========================

class CancerClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.base_model = models.densenet121(weights=None)

        num_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_trained_model():
    model = CancerClassifier(num_classes=2)

    model_path = Path(__file__).parent / "best_model_fine_tuning.pth"
    checkpoint = torch.load(model_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

model = load_trained_model()
st.success("Model loaded successfully!")


# =========================
# Import preprocessing
# =========================
from Preprocessing_pipeline import custom_preprocess

IMG_SIZE = (224, 224)

# =========================
# Utility Functions
# =========================

def read_image(uploaded_file):
    if uploaded_file.name.lower().endswith(".dcm"):
        dicom = pydicom.dcmread(uploaded_file)
        img = dicom.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
    else:
        img = np.array(Image.open(uploaded_file).convert("RGB"))
    return img


def image_quality_check(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)

    warnings = []
    if blur_score < 50:
        warnings.append("Image appears blurry")
    if brightness < 60 or brightness > 200:
        warnings.append("Poor lighting detected")

    return warnings


def prepare_image(img):
    img = cv2.resize(img, (224, 224))
    img = custom_preprocess(img)

    img = torch.tensor(img, dtype=torch.float32)
    img = img.permute(2, 0, 1)  # HWC → CHW
    img = img.unsqueeze(0)      # Add batch dim

    return img


import torch.nn.functional as F

def predict(img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)          # shape: [1, 2]
        probs = F.softmax(outputs, dim=1)    # convert logits → probabilities

        cancer_prob = probs[0][1].item()     # class index 1 = Cancer
        normal_prob = probs[0][0].item()

    if cancer_prob >= 0.5:
        return "Cancer", cancer_prob
    else:
        return "Normal", normal_prob


# =========================
# PDF Report
# =========================

def generate_pdf(image, heatmap, label, confidence):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"<b>{APP_NAME} Diagnostic Report</b>", styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"Prediction: <b>{label}</b>", styles['Normal']))
    elements.append(Paragraph(f"Confidence Score: {confidence*100:.2f}%", styles['Normal']))

    prompt = (
        "You should see a doctor for confirmation." if label == "Cancer"
        else "You don’t need to worry based on this scan."
    )
    elements.append(Paragraph(f"AI Note: {prompt}", styles['Italic']))
    elements.append(Spacer(1, 0.2 * inch))

    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    Image.fromarray(image).save(img_path)
    elements.append(RLImage(img_path, width=4*inch, height=4*inch))

    if heatmap is not None:
        hm_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        Image.fromarray(heatmap).save(hm_path)
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("Grad-CAM Heatmap", styles['Heading2']))
        elements.append(RLImage(hm_path, width=4*inch, height=4*inch))

    doc.build(elements)
    return tmp.name


# =========================
# UI Layout
# =========================

col1, col2 = st.columns([1, 5])
with col1:
    st.image(LOGO_PATH, width=120)
with col2:
    st.markdown(f"## {APP_NAME}")
    st.caption(TAGLINE)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a Chest X-ray (PNG, JPG, JPEG, DICOM)",
    type=["png", "jpg", "jpeg", "dcm"]
)

if uploaded_file:
    image = read_image(uploaded_file)
    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    warnings = image_quality_check(image)
    if warnings:
        st.warning("Image Quality Warning: " + ", ".join(warnings))

    with st.spinner("Analyzing..."):
        img_tensor = prepare_image(image)
        label, confidence = predict(img_tensor)

    st.subheader("Prediction Result")
    st.metric(label="Diagnosis", value=label)
    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

    if label == "Cancer":
        st.error("You should see a doctor for confirmation.")
        #heatmap = generate_gradcam(model, img_tensor)
        #overlay = overlay_heatmap(cv2.resize(image, IMG_SIZE), heatmap)
        #st.subheader("Grad-CAM Heatmap")
        #st.image(overlay, use_container_width=True)
    else:
        st.success("You don’t need to worry based on this scan.")
        heatmap = None
        overlay = None

    pdf_path = generate_pdf(image, overlay, label, confidence)
    with open(pdf_path, "rb") as f:
        st.download_button("Download PDF Report", f, file_name="RayScan_Report.pdf")

st.divider()
st.caption(
    "Disclaimer: Chest X-ray images are limited in detecting lung cancer. "
    "CT scans are most commonly used. RayScan is an early detection aid and "
    "not a tool for definitive diagnosis or medical decision-making."
)