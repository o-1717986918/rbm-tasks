#!/usr/bin/env python3
"""RoboMaster armor video inference with optional lightweight digit ID."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from digit_cnn import DigitClassifier, TemporalDigitVoting, estimate_light_color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--digit-model")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--digit-conf", type=float, default=0.65)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-output-width", type=int, default=1280)
    args = parser.parse_args()

    model = YOLO(args.weights)
    voter = None
    if args.digit_model:
        voter = TemporalDigitVoting(DigitClassifier(args.digit_model, args.digit_conf))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = min(width, args.max_output_width)
    out_height = round(height * out_width / width)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height))
    if not writer.isOpened():
        raise SystemExit(f"cannot create video: {args.out}")

    frames = detections = 0
    inference_ms = []
    while True:
        ok, original = cap.read()
        if not ok:
            break
        frame = cv2.resize(original, (out_width, out_height), interpolation=cv2.INTER_AREA) if (width, height) != (out_width, out_height) else original
        start = time.perf_counter()
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                               device=args.device, max_det=100, verbose=False)[0]
        inference_ms.append((time.perf_counter() - start) * 1000)
        boxes = [tuple(map(int, row)) for row in result.boxes.xyxy.cpu().numpy()]
        scores = result.boxes.conf.cpu().numpy().tolist()
        digit_outputs = voter.update(boxes, frame) if voter else [("?", 0.0)] * len(boxes)

        for (x1, y1, x2, y2), score, (digit, digit_score) in zip(boxes, scores, digit_outputs):
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(out_width - 1, x2), min(out_height - 1, y2)
            crop = frame[y1:y2, x1:x2]
            color_name = estimate_light_color(crop)
            color = (40, 40, 255) if color_name == "red" else (255, 120, 40) if color_name == "blue" else (0, 220, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"armor {score:.2f}"
            if voter:
                tag += f" {color_name} ID:{digit} {digit_score:.2f}"
            scale = max(0.45, min(0.8, (x2 - x1) / 180))
            cv2.putText(frame, tag, (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
            detections += 1
        writer.write(frame); frames += 1

    cap.release(); writer.release()
    mean_ms = sum(inference_ms) / max(1, len(inference_ms))
    print(f"frames={frames} detections={detections} mean_pipeline_ms={mean_ms:.2f} output={args.out}")


if __name__ == "__main__":
    main()
