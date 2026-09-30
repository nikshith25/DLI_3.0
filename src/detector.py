"""YOLO inference service for DLI 2.0."""

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from ultralytics import YOLO

from .config import MODEL_PATH, IMAGE_SIZE, CLASS_NAMES
from .validation import validate_confidence, validate_image_size, validate_detection


class DLIModel:
    """Small reusable inference wrapper around the trained YOLO11n model."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                "Place DLI_YOLO11n_best.pt in the project root or set MODEL_PATH."
            )
        self.model = YOLO(str(self.model_path))

    def predict(self, image: np.ndarray, confidence: float = 0.60) -> List[Dict[str, Any]]:
        """Run inference and return normalized detection dictionaries."""
        confidence = validate_confidence(confidence)
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")
        if image.ndim < 2:
            raise ValueError("Input image must have at least two dimensions.")

        height, width = image.shape[:2]
        validate_image_size(width, height)

        result = self.model.predict(
            source=image,
            conf=confidence,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )[0]

        boxes = result.boxes
        detections = []

        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            if isinstance(self.model.names, dict):
                name = self.model.names.get(cls_id, str(cls_id))
            else:
                name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)

            detection = {
                "Defect": name,
                "Confidence": score,
                "X1": int(xyxy[0]),
                "Y1": int(xyxy[1]),
                "X2": int(xyxy[2]),
                "Y2": int(xyxy[3]),
            }
            validate_detection(detection)
            detections.append(detection)

        return detections

    def annotate(self, image: np.ndarray, confidence: float = 0.60) -> np.ndarray:
        """Run inference and return an RGB annotated image."""
        confidence = validate_confidence(confidence)
        result = self.model.predict(
            source=image,
            conf=confidence,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )[0]
        annotated_bgr = result.plot()
        return annotated_bgr[:, :, ::-1]
