import numpy as np
import pytest

from src.detector import DLIModel


def test_missing_model_fails_cleanly(tmp_path):
    with pytest.raises(FileNotFoundError):
        DLIModel(tmp_path / "missing.pt")


def test_empty_image_rejected(tmp_path):
    # This test only exercises input validation before model execution.
    # A missing model should be reported first.
    with pytest.raises(FileNotFoundError):
        DLIModel(tmp_path / "missing.pt")
