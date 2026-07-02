from __future__ import annotations
import cv2
import numpy as np
from utils.preprocessing import splitPatches

def blockDifference(gray: np.ndarray, block: int = 8,) -> float: # Measure discontinuities across JPEG block boundaries.
    h, w = gray.shape
    values = []
    for x in range(block, w, block):
        diff = np.abs(gray[:, x].astype(np.float32) - gray[:, x - 1].astype(np.float32))
        values.append(diff.mean())

    for y in range(block, h, block):
        diff = np.abs(gray[y, :].astype(np.float32) - gray[y - 1, :].astype(np.float32))
        values.append(diff.mean())

    if len(values) == 0:
        return 0.0

    return float(np.mean(values))

def patchNoise(gray: np.ndarray,) -> float: # Estimate local noise level.
    patches = splitPatches(gray, rows=6, cols=6,)
    noise = []
    for patch in patches:
        blur = cv2.GaussianBlur(patch, (3, 3), 0,)
        diff = cv2.absdiff(patch, blur,)
        noise.append(diff.mean())
    return float(np.mean(noise))

def ringingScore(gray: np.ndarray,) -> float: # Estimate ringing around edges.
    edges = cv2.Canny(gray, 100, 200,)
    lap = cv2.Laplacian(gray, cv2.CV_64F,)
    values = np.abs(lap[edges > 0])

    if len(values) == 0:
        return 0.0

    return float(values.mean())

def dctEnergy(gray: np.ndarray,) -> float: # High-frequency DCT energy.
    img = gray.astype(np.float32)
    dct = cv2.dct(img)
    h, w = dct.shape
    roi = dct[h//2:, w//2:,]
    return float(np.mean(np.abs(roi)))


def compressionEvidence(gray: np.ndarray,):
    block = blockDifference(gray)
    noise = patchNoise(gray)
    ringing = ringingScore(gray)
    dct = dctEnergy(gray)
    confidence = np.mean([min(block / 20, 1), min(noise / 10, 1), min(ringing / 40, 1), min(dct / 50, 1),])

    return {
        "confidence": float(confidence),
        "features": {
            "block_difference": float(block),
            "patch_noise": float(noise),
            "ringing_score": float(ringing),
            "dct_energy": float(dct),
        },
    }