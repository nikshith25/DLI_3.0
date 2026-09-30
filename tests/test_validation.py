import pytest

from src.validation import (
    validate_bbox,
    validate_confidence,
    validate_detection,
    validate_image_extension,
    validate_image_size,
)


def test_valid_confidence():
    assert validate_confidence(0.60) == 0.60


@pytest.mark.parametrize("value", [-0.01, 1.01, 2])
def test_invalid_confidence(value):
    with pytest.raises(ValueError):
        validate_confidence(value)


def test_valid_extension():
    validate_image_extension("pcb.jpg", [".jpg", ".png"])


def test_invalid_extension():
    with pytest.raises(ValueError):
        validate_image_extension("pcb.pdf", [".jpg", ".png"])


def test_valid_image_size():
    validate_image_size(640, 480)


def test_invalid_image_size():
    with pytest.raises(ValueError):
        validate_image_size(0, 480)


def test_valid_bbox():
    validate_bbox(10, 20, 100, 120)


def test_invalid_bbox():
    with pytest.raises(ValueError):
        validate_bbox(100, 20, 10, 120)


def test_valid_detection():
    validate_detection({
        "Defect": "short",
        "Confidence": 0.91,
        "X1": 10,
        "Y1": 20,
        "X2": 100,
        "Y2": 120,
    })


def test_invalid_detection():
    with pytest.raises(ValueError):
        validate_detection({
            "Defect": "short",
            "Confidence": 0.91,
            "X1": 100,
            "Y1": 20,
            "X2": 10,
            "Y2": 120,
        })
