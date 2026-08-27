#!/usr/bin/env python3
"""Evaluate a detector and persist Ultralytics metrics as JSON."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = YOLO(args.weights).val(data=args.data, split=args.split, imgsz=args.imgsz,
                                    batch=args.batch, device=args.device, conf=0.001,
                                    iou=0.7, plots=True, verbose=True)
    report = {
        "weights": str(args.weights), "data": str(args.data), "split": args.split,
        "imgsz": args.imgsz, "batch": args.batch,
        "metrics": {key: float(value) for key, value in result.results_dict.items()},
        "speed_ms_per_image": {key: float(value) for key, value in result.speed.items()},
        "fitness": float(result.fitness),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
