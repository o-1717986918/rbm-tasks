#!/usr/bin/env python3
"""OpenCV-DNN wrapper for the 32x32 RoboMaster armor digit model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["?", "1", "2", "3", "4", "5"]


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32).reshape(-1)
    values -= values.max()
    exp = np.exp(values)
    return exp / max(float(exp.sum()), 1e-9)


class DigitClassifier:
    def __init__(self, model: str | Path, threshold: float = 0.65):
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.threshold = threshold

    def predict_proba(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        blob = gray.astype(np.float32)[None, None] / 127.5 - 1.0
        self.net.setInput(blob)
        return softmax(self.net.forward())

    def recognize(self, crop: np.ndarray):
        probs = self.predict_proba(crop)
        index = int(probs.argmax())
        confidence = float(probs[index])
        if index == 0 or confidence < self.threshold:
            return "?", confidence, probs
        return CLASS_NAMES[index], confidence, probs


def estimate_light_color(crop: np.ndarray) -> str:
    """Estimate armor light color using saturated red/blue evidence."""
    if crop is None or crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 12) | (hsv[..., 0] > 168)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 100)).sum()
    blue = ((hsv[..., 0] > 88) & (hsv[..., 0] < 135) & (hsv[..., 1] > 80) & (hsv[..., 2] > 90)).sum()
    if max(red, blue) < max(4, crop.shape[0] * crop.shape[1] * 0.004):
        return "unknown"
    return "red" if red > blue * 1.15 else "blue" if blue > red * 1.15 else "unknown"


@dataclass
class Track:
    center: tuple[float, float]
    size: float
    missed: int = 0
    probabilities: deque = field(default_factory=lambda: deque(maxlen=7))


class TemporalDigitVoting:
    """Small nearest-centre tracker used only to stabilize per-frame digit IDs."""
    def __init__(self, classifier: DigitClassifier):
        self.classifier = classifier
        self.tracks: list[Track] = []

    def update(self, boxes: list[tuple[int, int, int, int]], frame: np.ndarray):
        for track in self.tracks:
            track.missed += 1
        outputs = []
        used: set[int] = set()
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            size = max(8.0, ((x2 - x1) * (y2 - y1)) ** 0.5)
            candidates = [(i, ((cx - t.center[0]) ** 2 + (cy - t.center[1]) ** 2) ** 0.5 / max(size, t.size))
                          for i, t in enumerate(self.tracks) if i not in used]
            match = min(candidates, key=lambda pair: pair[1]) if candidates else None
            if match and match[1] < 1.2:
                index = match[0]; track = self.tracks[index]; used.add(index)
                track.center = (cx, cy); track.size = size; track.missed = 0
            else:
                track = Track((cx, cy), size); self.tracks.append(track); used.add(len(self.tracks) - 1)
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            track.probabilities.append(self.classifier.predict_proba(crop))
            mean = np.mean(np.stack(track.probabilities), axis=0)
            idx = int(mean.argmax()); conf = float(mean[idx])
            digit = CLASS_NAMES[idx] if idx and conf >= self.classifier.threshold else "?"
            outputs.append((digit, conf))
        self.tracks = [track for track in self.tracks if track.missed <= 10]
        return outputs
