#!/usr/bin/env bash
set -euo pipefail
data="${1:-dataset/armor_detection/data.yaml}"
model="${2:-yolo11n.pt}"
yolo detect train model="$model" data="$data" imgsz=640 epochs=60 patience=12 \
  batch=32 device=0 workers=2 cache=False amp=True optimizer=AdamW \
  lr0=0.001 lrf=0.01 cos_lr=True close_mosaic=10 degrees=4 translate=0.12 \
  scale=0.65 perspective=0.0005 mixup=0.05 project=runs name=armor_det_640_v2 \
  exist_ok=True seed=260827 deterministic=True
