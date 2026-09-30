"""DLI 2.0 Streamlit prototype."""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.config import (
    CLASS_NAMES,
    DEFAULT_CONFIDENCE,
    MODEL_PATH,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.detector import DLIModel
from src.reporting import csv_bytes, detections_to_dataframe, inspection_summary
from src.validation import validate_confidence, validate_image_extension

st.set_page_config(
    page_title="DLI 2.0 | PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("DLI 2.0 — PCB Defect Detection")
st.caption("AI-based PCB defect detection and spatial localization using YOLO11n.")

@st.cache_resource
def load_detector():
    return DLIModel(MODEL_PATH)

try:
    detector = load_detector()
except Exception as exc:
    st.error(f"Unable to load the DLI model: {exc}")
    st.stop()

st.sidebar.header("Detection Settings")
confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=float(DEFAULT_CONFIDENCE),
    step=0.05,
)
confidence = validate_confidence(confidence)

st.sidebar.markdown("### Supported Defects")
for name in CLASS_NAMES:
    st.sidebar.write(f"• {name}")

uploaded_file = st.file_uploader(
    "Upload a PCB image",
    type=[x.lstrip(".") for x in SUPPORTED_IMAGE_EXTENSIONS],
)

if uploaded_file is None:
    st.info("Upload a PCB image to start inspection.")
    st.markdown(
        "**Pipeline:** PCB Image → YOLO11n → Defect Classification → "
        "Bounding-Box Localization → Inspection Report"
    )
    st.stop()

try:
    validate_image_extension(uploaded_file.name, SUPPORTED_IMAGE_EXTENSIONS)
    image = Image.open(uploaded_file).convert("RGB")
except Exception as exc:
    st.error(f"Invalid image: {exc}")
    st.stop()

image_array = np.array(image)

with st.spinner("Analyzing PCB image..."):
    detections = detector.predict(image_array, confidence)
    annotated = detector.annotate(image_array, confidence)

left, right = st.columns(2)
with left:
    st.subheader("Input PCB")
    st.image(image, use_container_width=True)

with right:
    st.subheader("Detected Defects")
    st.image(annotated, use_container_width=True)

df = detections_to_dataframe(detections)
summary = inspection_summary(df, confidence)

st.subheader("Inspection Summary")
m1, m2, m3 = st.columns(3)
m1.metric("Defects Detected", summary["defects_detected"])
m2.metric("Highest Confidence", f"{summary['highest_confidence']:.2%}")
m3.metric("Threshold", f"{summary['threshold']:.2f}")

if df.empty:
    st.warning("No defects detected at the selected confidence threshold.")
else:
    st.subheader("Localization Details")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Detection Report (CSV)",
        data=csv_bytes(df),
        file_name="DLI_detection_report.csv",
        mime="text/csv",
    )

st.caption(
    "DLI 2.0 | YOLO11n | DeepPCB | Six defect classes | "
    "Default operating threshold: 0.60"
)
