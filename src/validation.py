"""Validation helpers for DLI 2.0."""

from pathlib import Path
from typing import Iterable, Mapping


def validate_confidence(value: float) -> float:
    """Validate and return a confidence threshold in [0, 1]."""
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Confidence threshold must be between 0.0 and 1.0.")
    return value


def validate_image_extension(filename: str, supported: Iterable[str]) -> None:
    """Raise ValueError when a filename has an unsupported image extension."""
    if not filename or Path(filename).suffix.lower() not in {x.lower() for x in supported}:
        raise ValueError("Unsupported image format.")


def validate_image_size(width: int, height: int) -> None:
    """Validate positive image dimensions."""
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("Image dimensions must be positive.")


def validate_bbox(x1: float, y1: float, x2: float, y2: float) -> None:
    """Validate an XYXY bounding box."""
    if x1 < 0 or y1 < 0:
        raise ValueError("Bounding-box coordinates cannot be negative.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding-box maximum coordinates must exceed minimum coordinates.")


def validate_detection(detection: Mapping) -> None:
    """Validate the minimum structure of one detection record."""
    required = {"Defect", "Confidence", "X1", "Y1", "X2", "Y2"}
    missing = required - set(detection)
    if missing:
        raise ValueError(f"Detection is missing fields: {sorted(missing)}")
    validate_confidence(float(detection["Confidence"]))
    validate_bbox(
        float(detection["X1"]),
        float(detection["Y1"]),
        float(detection["X2"]),
        float(detection["Y2"]),
    )
