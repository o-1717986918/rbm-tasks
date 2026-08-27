#!/usr/bin/env python3
"""Report IoU@0.5 recall by apparent armor width and hard-negative FP rate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def bucket(width: float) -> str:
    return "far_small_<24px" if width < 24 else "middle_24-64px" if width < 64 else "near_large_>=64px"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    args = parser.parse_args()
    model = YOLO(args.weights)
    image_root = args.dataset / "images" / "test"
    label_root = args.dataset / "labels" / "test"
    totals, matched = Counter(), Counter()
    tp = fp = background_images = background_fp_images = 0
    images = sorted(p for p in image_root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    results = model.predict(images, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=0,
                            batch=32, max_det=100, verbose=False)
    for image_path, result in zip(images, results):
        image = cv2.imread(str(image_path)); height, width = image.shape[:2]
        gt = []
        label_path = label_root / f"{image_path.stem}.txt"
        for line in label_path.read_text().splitlines() if label_path.exists() else []:
            values = list(map(float, line.split()))
            if len(values) < 5: continue
            _, cx, cy, bw, bh = values[:5]
            gt.append([(cx - bw / 2) * width, (cy - bh / 2) * height,
                       (cx + bw / 2) * width, (cy + bh / 2) * height])
        gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
        pred = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        if not len(gt):
            background_images += 1
            if len(pred): background_fp_images += 1
            fp += len(pred); continue
        pairs = []
        matrix = iou_matrix(gt, pred)
        for gi in range(len(gt)):
            totals[bucket(gt[gi, 2] - gt[gi, 0])] += 1
        for gi, pi in zip(*np.where(matrix >= 0.5)):
            pairs.append((float(matrix[gi, pi]), int(gi), int(pi)))
        used_gt, used_pred = set(), set()
        for _, gi, pi in sorted(pairs, reverse=True):
            if gi in used_gt or pi in used_pred: continue
            used_gt.add(gi); used_pred.add(pi); matched[bucket(gt[gi, 2] - gt[gi, 0])] += 1
        tp += len(used_gt); fp += len(pred) - len(used_pred)
    report = {
        "settings": {"imgsz": args.imgsz, "confidence": args.conf, "nms_iou": args.iou, "match_iou": 0.5},
        "test_images": len(images), "ground_truth": sum(totals.values()), "true_positive": tp,
        "false_positive": fp, "precision": tp / max(1, tp + fp), "recall": tp / max(1, sum(totals.values())),
        "recall_by_apparent_width": {key: {"matched": matched[key], "total": totals[key],
                                                   "recall": matched[key] / max(1, totals[key])}
                                      for key in ("far_small_<24px", "middle_24-64px", "near_large_>=64px")},
        "background_images": background_images, "background_images_with_false_positive": background_fp_images,
        "background_false_positive_image_rate": background_fp_images / max(1, background_images),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
