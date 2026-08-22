#!/usr/bin/env bash
set -euo pipefail
DATA=${1:-armor_dataset/data.yaml}
yolo detect train model=yolo11n.pt data="$DATA" imgsz=640 epochs=40 patience=10 batch=16 device=0 workers=8 cache=ram amp=False optimizer=AdamW lr0=0.001 cos_lr=True close_mosaic=10 project=runs name=armor_yolo11n
