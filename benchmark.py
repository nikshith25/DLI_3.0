"""Benchmark DLI 2.0 inference latency.

Usage:
    python benchmark.py --images path/to/test/images --limit 20

The script reports model-load time and per-image inference latency.
It does not fabricate performance numbers; all values are measured at runtime.
"""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import DEFAULT_CONFIDENCE, IMAGE_SIZE, MODEL_PATH, SUPPORTED_IMAGE_EXTENSIONS
from src.detector import DLIModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Directory containing test images.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    args = parser.parse_args()

    image_dir = Path(args.images)
    paths = [
        p for p in sorted(image_dir.iterdir())
        if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ][: args.limit]

    if not paths:
        raise SystemExit("No supported images found.")

    start = time.perf_counter()
    detector = DLIModel(MODEL_PATH)
    load_ms = (time.perf_counter() - start) * 1000

    latencies = []
    for path in paths:
        image = np.array(Image.open(path).convert("RGB"))
        start = time.perf_counter()
        detector.predict(image, args.confidence)
        latencies.append((time.perf_counter() - start) * 1000)

    avg = statistics.mean(latencies)
    median = statistics.median(latencies)
    minimum = min(latencies)
    maximum = max(latencies)
    fps = 1000.0 / avg if avg else 0.0

    print("\nDLI 2.0 Inference Benchmark")
    print("=" * 32)
    print(f"Model:              {MODEL_PATH.name}")
    print(f"Images evaluated:   {len(paths)}")
    print(f"Image size:         {IMAGE_SIZE} target")
    print(f"Confidence:         {args.confidence:.2f}")
    print(f"Model load time:    {load_ms:.2f} ms")
    print(f"Average latency:    {avg:.2f} ms/image")
    print(f"Median latency:     {median:.2f} ms/image")
    print(f"Minimum latency:    {minimum:.2f} ms/image")
    print(f"Maximum latency:    {maximum:.2f} ms/image")
    print(f"Approx. throughput: {fps:.2f} images/sec")
    print("=" * 32)


if __name__ == "__main__":
    main()
