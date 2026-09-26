"""Streamlit dashboard for the RA image-processing and scoring workflow.

Run with: streamlit run app.py
Set RA_CLASSIFIER_WEIGHTS, RA_SEVERITY_WEIGHTS, and YOLO_WEIGHTS to trained files.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import streamlit as st
import torch

from models import (DenseNetBinaryClassifier, GradCAM, OrdinalSeverityClassifier,
                    load_checkpoint, ordinal_grade)
from preprocess import enhance_clahe
from yolo_roi import crop_joint_rois


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_WEIGHTS = os.getenv("RA_CLASSIFIER_WEIGHTS", "checkpoints/ra_binary.pt")
SEVERITY_WEIGHTS = os.getenv("RA_SEVERITY_WEIGHTS", "checkpoints/ra_severity.pt")
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "checkpoints/joint_yolov8.pt")
IMAGE_TRANSFORM = A.Compose([A.Resize(224, 224),
                             A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


@st.cache_resource
def get_models() -> tuple[DenseNetBinaryClassifier, OrdinalSeverityClassifier]:
    binary = load_checkpoint(DenseNetBinaryClassifier(pretrained=False), CLASSIFIER_WEIGHTS, DEVICE)
    severity = load_checkpoint(OrdinalSeverityClassifier(pretrained=False), SEVERITY_WEIGHTS, DEVICE)
    binary.to(DEVICE).eval()
    severity.to(DEVICE).eval()
    return binary, severity


def prepare_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    data = IMAGE_TRANSFORM(image=rgb)["image"]
    return torch.from_numpy(data.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)


def draw_detections(image: np.ndarray, rois: list[dict]) -> np.ndarray:
    canvas = image.copy()
    for roi in rois:
        x1, y1, x2, y2 = roi["bbox_xyxy"]
        label = f"{roi['class_name']} {roi['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 220, 60), 2)
        cv2.putText(canvas, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (30, 220, 60), 2, cv2.LINE_AA)
    return canvas


st.set_page_config(page_title="RA Radiograph Analysis", page_icon="🩻", layout="wide")
st.title("Digital Image Processing and Computer Vision System")
st.caption("Research support for radiograph analysis. Outputs require validated, trained weights and clinician review; they are not a diagnosis.")
uploaded = st.file_uploader("Upload hand or wrist X-ray", type=["png", "jpg", "jpeg"])

if uploaded:
    raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    original = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if original is None:
        st.error("The uploaded file could not be decoded as an image.")
        st.stop()
    enhanced = enhance_clahe(original)
    st.subheader("1. Digital image enhancement")
    left, right = st.columns(2)
    left.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    right.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), caption="CLAHE enhanced", use_container_width=True)

    st.subheader("2. Joint region detection")
    try:
        with tempfile.TemporaryDirectory(prefix="ra_xray_") as temp_dir:
            image_path = Path(temp_dir) / "uploaded_xray.png"
            if not cv2.imwrite(str(image_path), original):
                raise OSError("Could not stage uploaded image for ROI detection")
            rois = crop_joint_rois(image_path, model_path=YOLO_WEIGHTS, output_dir=Path(temp_dir) / "rois")
    except (FileNotFoundError, ImportError, ValueError, OSError) as exc:
        st.error(f"ROI detection is unavailable: {exc}")
        st.stop()
    st.image(cv2.cvtColor(draw_detections(original, rois), cv2.COLOR_BGR2RGB),
             caption="Detected wrist/MCP/PIP candidates", use_container_width=True)
    if not rois:
        st.warning("No joint regions were detected. The image was not scored; check detector weights and annotations.")
        st.stop()

    st.subheader("3. Automated feature extraction and binary classification")
    try:
        binary_model, severity_model = get_models()
    except (FileNotFoundError, RuntimeError, ImportError) as exc:
        st.error(f"Trained classifier weights are unavailable or incompatible: {exc}")
        st.stop()
    predictions = []
    for roi in rois:
        roi_bgr = enhance_clahe(roi["image"])
        tensor = prepare_tensor(roi_bgr)
        with torch.no_grad():
            probability = torch.sigmoid(binary_model(tensor))[0].item()
        predictions.append((probability, roi_bgr, tensor))
    best = max(predictions, key=lambda item: item[0])
    probability, selected_roi, selected_tensor = best
    st.metric("RA presence confidence (highest ROI)", f"{probability * 100:.1f}%")
    threshold = st.number_input("RA-positive decision threshold", min_value=0.0, max_value=1.0,
                                value=0.5, step=0.05, help="Set this from validation-set operating characteristics.")
    if probability < threshold:
        st.info("The selected ROI is below the configured threshold. Severity scoring is not shown.")
    else:
        st.subheader("4. Ordinal severity scores")
        with torch.no_grad():
            output = severity_model(selected_tensor)
            jsn = int(ordinal_grade(output["jsn_logits"])[0].item())
            erosion = int(ordinal_grade(output["erosion_logits"])[0].item())
        c1, c2 = st.columns(2)
        c1.metric("Joint Space Narrowing grade", f"{jsn} / 4")
        c2.metric("Bone Erosion grade", f"{erosion} / 5")

    st.subheader("5. Grad-CAM spatial explainability")
    try:
        cam = GradCAM(binary_model)
        heatmap = cam(selected_tensor, output_size=(224, 224))
        cam.close()
        overlay = GradCAM.overlay(selected_roi, heatmap)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Binary classifier activation overlay",
                 use_container_width=True)
    except (RuntimeError, ValueError) as exc:
        st.warning(f"Grad-CAM could not be generated: {exc}")
