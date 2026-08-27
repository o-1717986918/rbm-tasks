#!/usr/bin/env python3
"""Train and export a tiny RoboMaster armor digit classifier.

Classes are ``unknown, 1, 2, 3, 4, 5``.  The script combines the public
Number-Classifier-for-RoboMaster crop set with the HKUST ENTERPRIZE 2025
armor-pattern set.  Splits are made by sequence-like groups rather than by
individual images, reducing adjacent-frame leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


CLASS_NAMES = ["unknown", "1", "2", "3", "4", "5"]


@dataclass(frozen=True)
class Item:
    image: Path
    label: int
    source: str
    group: str
    split: str


def split_for(group: str) -> str:
    bucket = int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def collect(number_root: Path, hkust_root: Path) -> list[Item]:
    items: list[Item] = []
    for txt in sorted(number_root.glob("*.txt")):
        tokens = txt.read_text(errors="ignore").split()
        if len(tokens) < 2:
            continue
        present, digit = int(tokens[0]), int(tokens[1])
        if not present:
            label = 0
        elif 1 <= digit <= 5:
            label = digit
        else:
            continue
        image = txt.with_suffix(".jpg")
        if not image.exists():
            continue
        try:
            seq = int(txt.stem) // 50
        except ValueError:
            seq = txt.stem
        group = f"number:{seq}"
        items.append(Item(image, label, "number_classifier", group, split_for(group)))

    for folder in sorted(p for p in hkust_root.iterdir() if p.is_dir()):
        code = folder.name.upper()
        if len(code) < 2:
            continue
        symbol = code[1:]
        label = int(symbol) if symbol in {"1", "2", "3", "4", "5"} else 0
        for image in sorted(folder.glob("*.jpg")):
            prefix = image.stem.split("_", 1)[0]
            group = f"hkust:{prefix}"
            items.append(Item(image, label, "hkust_pattern", group, split_for(group)))
    return items


def preprocess(gray: np.ndarray, size: int = 32) -> np.ndarray:
    if gray is None or gray.size == 0:
        raise ValueError("empty image")
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def augment(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = gray.shape
    angle = float(rng.uniform(-8, 8))
    scale = float(rng.uniform(0.88, 1.12))
    tx, ty = float(rng.uniform(-2.5, 2.5)), float(rng.uniform(-2.5, 2.5))
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[:, 2] += (tx, ty)
    gray = cv2.warpAffine(gray, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.35:
        k = int(rng.choice([3, 5]))
        gray = cv2.GaussianBlur(gray, (k, k), float(rng.uniform(0.1, 1.1)))
    if rng.random() < 0.45:
        alpha = float(rng.uniform(0.72, 1.28))
        beta = float(rng.uniform(-28, 28))
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    if rng.random() < 0.20:
        noise = rng.normal(0, rng.uniform(2, 10), gray.shape)
        gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return gray


class DigitDataset(Dataset):
    def __init__(self, items: list[Item], train: bool):
        self.items = items
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        gray = cv2.imread(str(item.image), cv2.IMREAD_GRAYSCALE)
        gray = preprocess(gray)
        if self.train:
            gray = augment(gray, np.random.default_rng(torch.initial_seed() + index))
        tensor = torch.from_numpy(gray).unsqueeze(0).float().div_(255.0)
        tensor = tensor.sub_(0.5).div_(0.5)
        return tensor, item.label


class TinyDigitNet(nn.Module):
    """Small static-shape CNN suitable for OpenCV DNN or TensorRT."""

    def __init__(self, classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.15), nn.Linear(64 * 4 * 4, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    confusion = np.zeros((6, 6), dtype=np.int64)
    loss_sum = 0.0
    count = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += float(criterion(logits, labels)) * len(labels)
        pred = logits.argmax(1)
        for truth, guess in zip(labels.cpu().numpy(), pred.cpu().numpy()):
            confusion[truth, guess] += 1
        count += len(labels)
    accuracy = float(np.trace(confusion) / max(1, confusion.sum()))
    recalls = np.diag(confusion) / np.maximum(1, confusion.sum(axis=1))
    return loss_sum / max(1, count), accuracy, recalls.tolist(), confusion.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number-root", type=Path, required=True)
    parser.add_argument("--hkust-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(260827); np.random.seed(260827); torch.manual_seed(260827)
    args.output.mkdir(parents=True, exist_ok=True)
    items = collect(args.number_root, args.hkust_root)
    by_split = {s: [x for x in items if x.split == s] for s in ("train", "val", "test")}
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["image", "label", "class_name", "source", "group", "split"])
        for item in items:
            writer.writerow([item.image, item.label, CLASS_NAMES[item.label], item.source, item.group, item.split])

    for split, rows in by_split.items():
        print(split, len(rows), dict(sorted(Counter(x.label for x in rows).items())))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_labels = [x.label for x in by_split["train"]]
    counts = Counter(train_labels)
    weights = [1.0 / counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(DigitDataset(by_split["train"], True), batch_size=args.batch,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(DigitDataset(by_split["val"], False), batch_size=args.batch,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(DigitDataset(by_split["test"], False), batch_size=args.batch,
                             shuffle=False, num_workers=2, pin_memory=True)

    model = TinyDigitNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward(); optimizer.step()
            total_loss += float(loss) * len(labels); seen += len(labels)
        scheduler.step()
        val_loss, val_acc, recalls, confusion = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss / seen, "val_loss": val_loss,
               "val_accuracy": val_acc, "val_recall": recalls}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        score = val_acc + 0.05 * min(recalls[1:])
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "classes": CLASS_NAMES, "input_size": 32,
                        "epoch": epoch, "val_accuracy": val_acc}, args.output / "armor_digit_tiny_best.pt")

    checkpoint = torch.load(args.output / "armor_digit_tiny_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_acc, test_recalls, test_confusion = evaluate(model, test_loader, device)
    metrics = {"classes": CLASS_NAMES, "parameters": sum(p.numel() for p in model.parameters()),
               "input": "1x1x32x32 grayscale, value=(pixel/255-0.5)/0.5",
               "dataset_total": len(items),
               "split_counts": {s: dict(Counter(x.label for x in rows)) for s, rows in by_split.items()},
               "best_epoch": checkpoint["epoch"], "val_accuracy": checkpoint["val_accuracy"],
               "test_loss": test_loss, "test_accuracy": test_acc,
               "test_recall": test_recalls, "test_confusion": test_confusion,
               "history": history}
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    model = model.cpu().eval()
    torch.onnx.export(model, torch.zeros(1, 1, 32, 32), args.output / "armor_digit_tiny.onnx",
                      input_names=["images"], output_names=["logits"], opset_version=12,
                      dynamic_axes=None, do_constant_folding=True)
    print(json.dumps({"test_accuracy": test_acc, "test_recall": test_recalls,
                      "onnx": str(args.output / "armor_digit_tiny.onnx")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
