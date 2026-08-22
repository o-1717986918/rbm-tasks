#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_dataset.py
把 raw/<class>/ 下的图片 + 同名 YOLO txt，按类分层划分成 train/val，
复制到 datasets/cat_daodun/images/{train,val} + labels/{train,val}，并生成 data.yaml。

用法（WSL 项目内）：
  python scripts/split_dataset.py
可选：
  --raw <dir>            默认 <project>/raw
  --out <dir>            默认 <project>/datasets/cat_daodun
  --val-ratio 0.15       val 占比（图很少时每类至少留 1 张）
  --seed 0
标注约定：raw/<class>/img.jpg 与 raw/<class>/img.txt（YOLO 行式，同名同目录）。
只处理有同名 txt 的图；缺标签的图会跳过并计数。
"""
import argparse, os, random, shutil, sys
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
CLASSES = {          # 目录名 -> class id（与 data.yaml 一致）
    "miaocuijiao_cat": 0,
    "daodun": 1,
}
CLASSES_INV = {v: k for k, v in CLASSES.items()}


def find_images(class_dir):
    """返回 [(img_path, label_path)]，只保留有同名 txt 的图。"""
    pairs, missing = [], []
    for img in sorted(class_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        txt = img.with_suffix(".txt")
        if txt.exists():
            pairs.append((img, txt))
        else:
            missing.append(img)
    return pairs, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="datasets/cat_daodun")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    raw = Path(a.raw).resolve()
    out = Path(a.out).resolve()
    random.seed(a.seed)

    all_pairs = []   # (class_id, img, label)
    for cname, cid in CLASSES.items():
        cdir = raw / cname
        if not cdir.is_dir():
            print(f"[skip] {cname}: 目录 {cdir} 不存在")
            continue
        pairs, missing = find_images(cdir)
        if missing:
            print(f"[warn] {cname}: {len(missing)} 张图缺少同名 .txt，已跳过。例: {missing[0].name}")
        for img, txt in pairs:
            all_pairs.append((cid, img, txt))
        print(f"[collect] {cname}: {len(pairs)} 张带标注图")

    if not all_pairs:
        print("没有找到任何带标注的图，终止。请先在 raw 下放图并标好 YOLO txt。")
        sys.exit(1)

    # 按类分层划分 train/val
    train, val = [], []
    for cid in CLASSES.values():
        items = [p for p in all_pairs if p[0] == cid]
        random.shuffle(items)
        if len(items) >= 3:
            n_val = max(1, round(len(items) * a.val_ratio))
        elif len(items) == 2:
            n_val = 1
        else:
            n_val = 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    random.shuffle(train)

    # 清空并重建 out 的 train/val
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = out / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    def place(cid, img, txt, split, seen):
        cls = CLASSES_INV[cid]
        stem = img.stem
        if not stem.startswith(cls):                 # 已含类名前缀则不重复
            stem = f"{cls}_{stem}"
        dest_img = out / "images" / split / (stem + img.suffix.lower())
        dest_txt = out / "labels" / split / (stem + ".txt")
        if dest_img.name in seen:
            return False
        seen.add(dest_img.name)
        shutil.copy2(img, dest_img)
        shutil.copy2(txt, dest_txt)
        return True

    seen_train, seen_val = set(), set()
    for cid, img, txt in train:
        place(cid, img, txt, "train", seen_train)
    for cid, img, txt in val:
        place(cid, img, txt, "val", seen_val)

    # 按类统计实际 train/val 数量（从 label 内容反推）
    stats = defaultdict(lambda: defaultdict(int))
    for split in ("train", "val"):
        for txt in (out / "labels" / split).glob("*.txt"):
            first = txt.read_text().strip().split()
            if first and first[0].lstrip("-").isdigit():
                cid = int(first[0])
                cls = CLASSES_INV.get(cid, str(cid))
                stats[cls][split] += 1

    print("\n== 划分结果 ==")
    for cname in CLASSES:
        print(f"  {cname}: train={stats[cname]['train']}  val={stats[cname]['val']}")
    total_train = len(list((out / "labels" / "train").glob("*.txt")))
    total_val = len(list((out / "labels" / "val").glob("*.txt")))
    print(f"  [合计] train={total_train}  val={total_val}")

    (out / "data.yaml").write_text(
        "# YOLO 检测数据集配置（由 split_dataset.py 生成）\n"
        f"path: {out}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {cid}: {cname}\n" for cname, cid in CLASSES.items())
    )
    print(f"\n[ok] data.yaml -> {out / 'data.yaml'}")
    print("接下来：python scripts/check_dataset.py  然后 yolo train ...")


if __name__ == "__main__":
    main()
