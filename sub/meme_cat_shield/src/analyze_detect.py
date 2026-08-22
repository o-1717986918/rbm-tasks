#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析识别效果差的原因：对比 训练集(meme_det) 与 真实识别域(raw 封面)。"""
import cv2, glob, os, statistics
from pathlib import Path
from ultralytics import YOLO

P = Path("/home/win98/my_projects/yoloproj/rbmproject")
M = YOLO(str(P / "runs_meme/v1/weights/best.pt"))

print("=" * 60)
print("【1】训练集 meme_det 标签的 bbox 归一化分布(center/cx+cy, 宽高)")
print("=" * 60)
for split in ("train", "val"):
    for name in ("miaocuijiao_cat", "sword_shield_dog"):
        txts = sorted((P / "datasets/meme_det/labels" / split).glob(f"{name}_*.txt"))
        if not txts:
            continue
        cx, cy, w, h, nobj = [], [], [], [], []
        for t in txts:
            lines = [l.strip() for l in t.read_text().splitlines() if l.strip()]
            nobj.append(len(lines))
            for l in lines:
                parts = l.split()
                if len(parts) == 5:
                    cx.append(float(parts[1])); cy.append(float(parts[2]))
                    w.append(float(parts[3])); h.append(float(parts[4]))
        def stat(v):
            return f"mean={statistics.mean(v):.3f} min={min(v):.3f} max={max(v):.3f}"
        print(f"[{split}/{name}] 图={len(txts)} 标签目标数/图均值={statistics.mean(nobj):.2f}")
        print(f"     cx {stat(cx)}  cy {stat(cy)}")
        print(f"     宽 {stat(w)}  高 {stat(h)}")

print()
print("=" * 60)
print("【2】真实封面(raw) 上用模型识别的情况")
print("=" * 60)
for name in ("miaocuijiao_cat", "daodun"):
    imgs = sorted((P / "raw" / name).glob("*.jpg"))
    tot, hits, confs, box_ratio = 0, 0, [], []
    for im in imgs:
        frame = cv2.imread(str(im))
        H, W = frame.shape[:2]
        r = M.predict(frame, imgsz=640, conf=0.2, device=0, verbose=False)[0]
        tot += 1
        if r.boxes is not None and len(r.boxes) > 0:
            hits += 1
            for b in r.boxes:
                confs.append(float(b.conf))
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                box_ratio.append((x2 - x1) * (y2 - y1) / (W * H))
    print(f"[{name}] 图={tot}  检出框>0 的图={hits}  ({hits/tot*100:.0f}%)")
    if confs:
        print(f"     平均置信度={statistics.mean(confs):.3f}  框占画面比例均值={statistics.mean(box_ratio):.3f}")

print()
print("=" * 60)
print("【3】训练图 vs 真实图 尺寸/宽高比")
print("=" * 60)
import glob as g
img0 = cv2.imread(str((P / "datasets/meme_det/images/train/miaocuijiao_cat_001.jpg")))
raw0 = cv2.imread(str((P / "raw/miaocuijiao_cat/miaocuijiao_cat_001.jpg")))
print(f"训练图(meme_det): {img0.shape[1]}x{img0.shape[0]}  aspect={img0.shape[1]/img0.shape[0]:.2f}")
print(f"真实图(raw 封面): {raw0.shape[1]}x{raw0.shape[0]}  aspect={raw0.shape[1]/raw0.shape[0]:.2f}")
