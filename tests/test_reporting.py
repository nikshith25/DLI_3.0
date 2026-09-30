from src.reporting import detections_to_dataframe, inspection_summary, csv_bytes


def test_dataframe_schema():
    df = detections_to_dataframe([])
    assert list(df.columns) == ["Defect", "Confidence", "X1", "Y1", "X2", "Y2"]


def test_dataframe_and_summary():
    detections = [{
        "Defect": "open",
        "Confidence": 0.95,
        "X1": 10,
        "Y1": 20,
        "X2": 50,
        "Y2": 60,
    }]
    df = detections_to_dataframe(detections)
    summary = inspection_summary(df, 0.60)

    assert len(df) == 1
    assert summary["defects_detected"] == 1
    assert summary["highest_confidence"] == 0.95
    assert b"open" in csv_bytes(df)
