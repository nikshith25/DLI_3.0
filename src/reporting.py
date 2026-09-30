"""Reporting utilities for DLI 2.0."""

from io import BytesIO
from typing import List, Dict, Any

import pandas as pd


COLUMNS = ["Defect", "Confidence", "X1", "Y1", "X2", "Y2"]


def detections_to_dataframe(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert detections to a stable tabular representation."""
    df = pd.DataFrame(detections, columns=COLUMNS)
    if not df.empty:
        df["Confidence"] = df["Confidence"].astype(float).round(4)
        for col in ["X1", "Y1", "X2", "Y2"]:
            df[col] = df[col].astype(int)
    return df


def csv_bytes(df: pd.DataFrame) -> bytes:
    """Return UTF-8 CSV bytes suitable for a download button."""
    return df.to_csv(index=False).encode("utf-8")


def inspection_summary(df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    """Create a compact inspection summary."""
    highest = float(df["Confidence"].max()) if not df.empty else 0.0
    return {
        "defects_detected": int(len(df)),
        "highest_confidence": highest,
        "threshold": float(threshold),
    }
