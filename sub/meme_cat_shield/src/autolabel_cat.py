#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autolabel_cat.py — 半自动标注
- 妙脆角猫：用 COCO 预训练 yolov8n 检测"猫"(class=15)，取置信度最高的一只，
  生成 class=0 的 bbox 到同名 .txt。
- 刀盾：COCO 不认识，生成空 .txt 占位，供人工标注(class=1)。
标注后可人工微调（妙脆角猫框扩大到含妙脆角；刀盾手绘）。
"""
import cv2
from pathlib import Path
from ultralytics import YOLO

ROOT = Path("/home/win98/my_projects/yoloproj/rbmproject")
CAT = ROOT / "raw" / "miaocuijiao_cat"
DD = ROOT / "raw" / "daodun"
CAT_CLS = 15     # COCO cat

m = YOLO(str(ROOT / "yolov8n.pt"))
print("COCO names[15] =", m.names.get(CAT_CLS))

auto, empty = 0, 0
for img in sorted(CAT.glob("*.jpg")):
    frame = cv2.imread(str(img))
    if frame is None:
        print("[warn] 无法读取", img.name); continue
    h, w = frame.shape[:2]
    r = m.predict(frame, conf=0.3, device=0, verbose=False)[0]
    best, bc = None, 0.0
    for b in r.boxes:
        if int(b.cls) == CAT_CLS:
            c = float(b.conf)
            if c > bc:
                bc, best = c, b
    txt = img.with_suffix(".txt")
    if best is not None:
        x1, y1, x2, y2 = [float(v) for v in best.xyxy[0]]
        line = f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n"
        txt.write_text(line)
        auto += 1
    else:
        txt.write_text("")
        empty += 1
print(f"妙脆角猫: 自动框出 {auto} 张, 空(需人工) {empty} 张")

for img in sorted(DD.glob("*.jpg")):
    (img.with_suffix(".txt")).write_text("")
print(f"刀盾: 已生成空 txt {len(list(DD.glob('*.jpg')))} 个，请人工标注(class=1)")
