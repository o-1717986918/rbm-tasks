#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo.py — 加载检测模型 best.pt，对图片/视频推理，输出"中文标签 + 检测框"画面。

特色：
  - 中文标签（妙脆角猫 / 刀盾）用 PIL 手绘 + 自定义中文字体，避免 cv2/ultralytics 的 CJK tofu。
  - 支持单张图、目录、视频；模型保留 ASCII names，中文只在显示层映射。
  - 输出到 <out>/ 下；视频写为 mp4。

用法（WSL 内，conda activate yolo_env）：
  python scripts/demo.py --source 图片或视频或目录 \
      --model runs_det/v1/weights/best.pt \
      --out outputs/demo --conf 0.25 --imgsz 640 --device 0
"""
import argparse, os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
CN = {0: "妙脆角猫", 1: "刀盾"}
COLOR = {0: (60, 120, 255), 1: (80, 200, 255)}   # RGB（B 类橙黄、A 类蓝）


def load_font(pil_img, base=32):
    size = max(24, base)
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def annotate(img_bgr, results, base=32, conf_threshold=0.2):
    """在 BGR 图上画中文标签+框，返回 BGR 图。只画 conf>conf_threshold 的目标。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    # 字号随图像尺寸自适应，保证置信度清晰可见
    font = load_font(pil, max(30, int(min(pil.size) / 14)))
    for r in results:
        boxes = getattr(r.boxes, "xyxy", None)
        if boxes is None:
            continue
        for i in range(len(boxes)):
            x0, y0, x1, y1 = [int(v) for v in boxes[i].tolist()]
            cls = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i]) if r.boxes.conf is not None else 0.0
            if conf <= conf_threshold:          # 只显示 >0.2 的目标
                continue
            color = COLOR.get(cls, (255, 0, 0))
            text = f"{CN.get(cls, str(cls))} {conf:.2f}"
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            # 文字置于框上缘，带底色
            tb = draw.textbbox((x0, y0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = y0 - th - 8 if (y0 - th - 8) > 0 else y0 + 4
            draw.rectangle([x0, ty - 2, x0 + tw + 6, ty + th + 2], fill=color)
            draw.text((x0 + 3, ty), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="runs_det/v1/weights/best.pt")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.model)
    src = a.source
    if os.path.isdir(src):
        imgs = [f for f in sorted(os.listdir(src))
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        for name in imgs:
            p = os.path.join(src, name)
            frame = cv2.imread(p)
            if frame is None:
                print(f"[skip] 无法读取 {p}")
                continue
            results = model.predict(frame, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)
            out = annotate(frame, results, conf_threshold=a.conf)
            dst = os.path.join(a.out, os.path.splitext(name)[0] + "_demo.jpg")
            cv2.imwrite(dst, out)
            print(f"[ok] {p} -> {dst}")
        return

    # 单文件：按扩展名区分视频 / 图片
    VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg"}
    if src.lower().endswith(tuple(VIDEO_EXT)):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[error] 无法打开视频 {src}")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # mp4v 编码要求帧尺寸是 16 的倍数
        ow = (w + 15) // 16 * 16
        oh = (h + 15) // 16 * 16
        name = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(a.out, name + "_demo.mp4")
        writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (ow, oh))
            results = model.predict(frame, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)
            writer.write(annotate(frame, results, conf_threshold=a.conf))
            n += 1
        cap.release()
        writer.release()
        print(f"[ok] 视频 {src} -> {dst} (共 {n} 帧)")
        return

    # 图片
    frame = cv2.imread(src)
    if frame is None:
        print(f"[error] 无法读取 {src}")
        sys.exit(1)
    results = model.predict(frame, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)
    out = annotate(frame, results, conf_threshold=a.conf)
    dst = os.path.join(a.out, os.path.splitext(os.path.basename(src))[0] + "_demo.jpg")
    cv2.imwrite(dst, out)
    print(f"[ok] {src} -> {dst}")


if __name__ == "__main__":
    main()
