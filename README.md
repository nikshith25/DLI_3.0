# DLI 2.0 — Defect Localisation Intelligence

## AI-Based PCB Defect Detection and Localization

DLI 2.0 is a computer-vision prototype for automated PCB defect detection and spatial localization. The current implementation uses a YOLO11n model trained on the DeepPCB dataset.

### Current implemented scope

- PCB image upload
- Six-class defect detection:
  - open
  - short
  - mousebite
  - spur
  - copper
  - pin-hole
- Bounding-box localization
- Confidence scoring and threshold filtering
- Inspection summary
- CSV detection report
- Streamlit deployment
- Automated input/output validation
- Automated tests
- Runtime inference benchmarking

### Architecture

```text
PCB Image
   |
   v
Input Validation
   |
   v
YOLO11n Detector
   |
   +--> Defect Class
   +--> Confidence
   +--> Bounding Box
   |
   v
Confidence Filtering
   |
   v
Localization + Visualization
   |
   v
Inspection Report / CSV
```

### Evaluation

The evaluated DeepPCB test set produced the following result at confidence threshold 0.60:

| Metric | Result |
|---|---:|
| Precision | 97.64% |
| Recall | 96.54% |
| F1-score | 97.09% |

These are the project's reported evaluation results; benchmark latency should be measured on the target runtime using `benchmark.py` rather than assumed.

### Installation

```bash
python -m pip install -r requirements.txt
```

### Run the application

```bash
python -m streamlit run streamlit_app.py
```

### Run tests

```bash
pytest -q
```

### Run inference benchmark

Example:

```bash
python benchmark.py --images /path/to/DeepPCB_YOLO/test/images --limit 20
```

The benchmark measures model loading time and inference latency on the machine where it is executed.

### Configuration

Copy `.env.example` to `.env` when local environment configuration is needed.

Do not commit `.env` or API keys/secrets.

### SDG 9 alignment

DLI 2.0 supports **SDG 9: Industry, Innovation and Infrastructure** by applying computer vision and deep learning to automated industrial PCB quality inspection. The current prototype focuses specifically on defect detection and localization; severity estimation, root-cause analysis, and corrective recommendations remain future extensions.

### Future research directions

- Defect severity estimation
- Explainable AI
- Root-cause analysis
- Corrective-action recommendation
- Multi-dataset and real-world validation
- Real-time production-line inspection
- Edge deployment

### Project structure

```text
DLI_2.0/
├── streamlit_app.py
├── benchmark.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── DLI_YOLO11n_best.pt
├── src/
│   ├── config.py
│   ├── detector.py
│   ├── validation.py
│   └── reporting.py
└── tests/
    ├── test_validation.py
    ├── test_detection.py
    └── test_reporting.py
```
