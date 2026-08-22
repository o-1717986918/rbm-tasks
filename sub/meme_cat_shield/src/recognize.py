#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recognize.py — 纯 Python 识别脚本
加载训练好的检测模型，对图片/目录/视频推理，
打印每个目标的【中文类别、置信度、bbox 坐标】，并可选保存带框标注图。

用法:
  python recognize.py --source <图/目录/视频> --model runs_meme/v1/weights/best.pt \
      --conf 0.2 --imgsz 640 --device 0 [--save --out outputs/rec]
"""
import argparse, os, sys, json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

CN = {0: "妙脆角猫", 1: "刀盾"}
COLOR = {0: (60, 120, 255), 1: (80, 200, 255)}
FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg"}


def get_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def draw_marked(img_bgr, results, conf):
    """在 BGR 图上画中文标签+框，返回 BGR 图。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = get_font(max(30, int(min(pil.size) / 14)))
    if results.boxes is not None:
        for i in range(len(results.boxes)):
            x1, y1, x2, y2 = [float(v) for v in results.boxes.xyxy[i]]
            cls = int(results.boxes.cls[i])
            c = float(results.boxes.conf[i]) if results.boxes.conf is not None else 0.0
            if c <= conf:
                continue
            color = COLOR.get(cls, (255, 0, 0))
            text = f"{CN.get(cls, str(cls))} {c:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            tb = draw.textbbox((x1, y1), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = y1 - th - 8 if (y1 - th - 8) > 0 else y1 + 4
            draw.rectangle([x1, ty - 2, x1 + tw + 6, ty + th + 2], fill=color)
            draw.text((x1 + 3, ty), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def recognize(frame, model, conf, imgsz, device):
    """推理一帧，返回识别结果列表。"""
    results = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    dets = []
    if results.boxes is not None:
        for i in range(len(results.boxes)):
            x1, y1, x2, y2 = [float(v) for v in results.boxes.xyxy[i]]
            cls = int(results.boxes.cls[i])
            c = float(results.boxes.conf[i]) if results.boxes.conf is not None else 0.0
            if c <= conf:
                continue
            dets.append({
                "class": CN.get(cls, str(cls)),
                "cls_id": cls,
                "confidence": round(c, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })
    return dets, results


def main():
    ap = argparse.ArgumentParser(description="纯 Python 识别脚本")
    ap.add_argument("--source", required=True, help="图片 / 目录 / 视频")
    ap.add_argument("--model", default="runs_meme/v1/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    ap.add_argument("--save", action="store_true", help="保存带框标注图/视频")
    ap.add_argument("--out", default="outputs/rec")
    a = ap.parse_args()

    model = YOLO(a.model)
    src = a.source

    def handle_image(img_path, out_dir):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[skip] 无法读取 {img_path}")
            return
        dets, results = recognize(frame, model, a.conf, a.imgsz, a.device)
        print(f"\n== {os.path.basename(img_path)} ==")
        if not dets:
            print("  (无目标)")
        for d in dets:
            print(f"  [{d['class']}] confidence={d['confidence']}  bbox={d['bbox']}")
        if a.save and dets:
            os.makedirs(out_dir, exist_ok=True)
            marked = draw_marked(frame, results, a.conf)
            dst = os.path.join(out_dir, os.path.splitext(os.path.basename(img_path))[0] + "_rec.jpg")
            cv2.imwrite(dst, marked)
            print(f"  [saved] {dst}")

    if os.path.isdir(src):
        imgs = [f for f in sorted(os.listdir(src))
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        for name in imgs:
            handle_image(os.path.join(src, name), a.out)
        return

    if src.lower().endswith(tuple(VIDEO_EXT)):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[error] 无法打开视频 {src}"); sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + 15) // 16 * 16
        h = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 15) // 16 * 16
        os.makedirs(a.out, exist_ok=True)
        name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(a.out, name + "_rec.mp4")
        writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if a.save else None
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (w, h))
            dets, results = recognize(frame, model, a.conf, a.imgsz, a.device)
            n += 1
            print(f"\r帧 {n}: {len(dets)} 个目标", end="")
            if a.save and writer is not None:
                writer.write(draw_marked(frame, results, a.conf))
        print()
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[saved] {dst}")
        return

    handle_image(src, a.out)


if __name__ == "__main__":
    main()
