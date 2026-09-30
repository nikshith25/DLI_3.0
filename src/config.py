"""Central configuration for DLI 2.0."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "DLI_YOLO11n_best.pt"))
DEFAULT_CONFIDENCE = float(os.getenv("DEFAULT_CONFIDENCE", "0.60"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.50"))

CLASS_NAMES = (
    "open",
    "short",
    "mousebite",
    "spur",
    "copper",
    "pin-hole",
)

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
