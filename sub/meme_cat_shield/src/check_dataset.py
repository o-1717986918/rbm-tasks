#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_dataset.py
对 datasets/cat_daodun 做训练前体检：
  - 图片-标签配对（缺 txt / 缺图）
  - 标签行格式与归一化坐标范围（越界框）
  - 空标签（警告）
  - 图片损坏（PIL 打不开）
  - md5 重复

用法（WSL 项目内）：
  python scripts/check_dataset.py            # 默认 datasets/cat_daodun
  python scripts/check_dataset.py --data <dir>
通过则打印 "[ok] 可以开始训练"；有问题输出警告/错误并标红。
"""
import argparse, hashlib
from collections import defaultdict
from pathlib import Path
from PIL import Image


def check_split(data, split, problems):
    data = Path(data)
    images_dir = data / "images" / split
    labels_dir = data / "labels" / split
    n_img = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
    n_lbl = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
    print(f"[{split}] images={n_img} labels={n_lbl}")
    return images_dir, labels_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/cat_daodun")
    a = ap.parse_args()
    data = Path(a.data).resolve()
    print(f"检查数据集: {data}\n" + "-" * 50)

    problems = []
    empty_labels = 0
    for split in ("train", "val", "test"):
        if not (data / "images" / split).exists() and split == "test":
            continue
        images_dir, labels_dir = check_split(data, split, problems)

        img_by_name = {}
        for im in images_dir.glob("*"):
            if im.is_file():
                img_by_name[im.stem] = im

        label_stems = {p.stem for p in labels_dir.glob("*.txt")}
        for stem, im in img_by_name.items():
            if stem not in label_stems:
                problems.append(f"[{split}] 图片 {im.name} 缺少同名标签文件")

        # 1) 缺标签 2) 缺图 3) 标签格式/范围
        lbl_count = 0
        for lbl in labels_dir.glob("*.txt"):
            lbl_count += 1
            img = img_by_name.get(lbl.stem)
            if img is None:
                problems.append(f"[{split}] 标签 {lbl.name} 无对应图片")
                continue
            text = lbl.read_text(errors="ignore").strip()
            if not text:
                # Empty labels are the standard YOLO representation for
                # background/hard-negative images.
                empty_labels += 1
                continue
            for ln, line in enumerate(text.splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    problems.append(f"[{split}] {lbl.name}:{ln} 行格式错（应为 class cx cy w h）: {line!r}")
                    continue
                try:
                    c, x, y, w, h = (float(p) for p in parts)
                except ValueError:
                    problems.append(f"[{split}] {lbl.name}:{ln} 非数值: {line!r}")
                    continue
                if not c.is_integer() or int(c) not in (0, 1):
                    problems.append(f"[{split}] {lbl.name}:{ln} 非法类别 {c}")
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    problems.append(f"[{split}] {lbl.name}:{ln} 坐标越界: {line!r}")
                eps = 2e-6  # tolerate six-decimal serialization at image edges
                if x - w / 2 < -eps or x + w / 2 > 1 + eps or y - h / 2 < -eps or y + h / 2 > 1 + eps:
                    problems.append(f"[{split}] {lbl.name}:{ln} 框超图片边界: {line!r}")

        # 4) 图片损坏检测（抽检比例：>200 抽 200，否则全检）
        n_to_check = min(len(img_by_name), 200)
        for i, (stem, im) in enumerate(img_by_name.items()):
            if i >= n_to_check:
                break
            try:
                with Image.open(im) as imo:
                    imo.verify()
            except Exception as e:
                problems.append(f"[{split}] 图片损坏 {im.name}: {e}")

    # 5) md5 重复（全局）
    seen, dups = {}, []
    for split in ("train", "val", "test"):
        for im in (data / "images" / split).glob("*"):
            if not im.is_file():
                continue
            h = hashlib.md5(im.read_bytes()).hexdigest()
            if h in seen:
                dups.append(f"{seen[h]} == {im.name}")
            else:
                seen[h] = im.name
    if dups:
        problems.append(f"检测到 {len(dups)} 组 md5 重复图：\n  " + "\n  ".join(dups[:20]))

    print(f"空标签/负样本: {empty_labels}")
    print("-" * 50)
    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for p in problems:
            print("  [!!] " + p)
        print("\n请先修复上述问题再训练。")
        raise SystemExit(1)
    else:
        print("[ok] 数据体检通过，可以开始训练。")


if __name__ == "__main__":
    main()
