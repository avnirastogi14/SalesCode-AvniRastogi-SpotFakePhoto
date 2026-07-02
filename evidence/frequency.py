from __future__ import annotations

import cv2
import numpy as np

from utils.preprocessing import equalizeHistogram


# FFT is used to analyse frequency.
def computeFFT(gray: np.ndarray) -> np.ndarray:  # Computes centered FFT magnitude spectrum.
    gray = equalizeHistogram(gray)
    gray = gray.astype(np.float32)

    fft = np.fft.fft2(gray)
    fft = np.fft.fftshift(fft)

    magnitude = np.abs(fft)
    magnitude = np.log1p(magnitude)

    return magnitude


def removeCenter(
    magnitude: np.ndarray,
    radius: int = 10,
) -> np.ndarray:

    mag = magnitude.copy()

    h, w = mag.shape

    cy = h // 2
    cx = w // 2

    cv2.circle(mag, (cx, cy), radius, 0, -1,)
    return mag


def fftPeakStrength(magnitude: np.ndarray,) -> float:
    mag = removeCenter(magnitude)

    peak = np.max(mag)
    mean = np.mean(mag)

    score = peak / (mean + 1e-6)

    return float(score)


def midFrequencyRatio(magnitude: np.ndarray,) -> float:
    h, w = magnitude.shape

    cy = h // 2
    cx = w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    rMaxi = r.max()

    low = magnitude[r < rMaxi * 0.15].sum()

    mid = magnitude[(r >= rMaxi * 0.15) & (r < rMaxi * 0.45)].sum()
    high = magnitude[r >= rMaxi * 0.45].sum()

    return float(mid/(low + high + 1e-6))


def radialE(magnitude: np.ndarray, bins: int = 25,) -> np.ndarray:
    # FFT's radial energy distribution

    h, w = magnitude.shape

    cy = h // 2
    cx = w // 2

    y, x = np.indices((h, w))

    rad = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rad = rad.astype(np.int32)
    rMaxi = rad.max()
    e = np.zeros(rMaxi + 1)

    for r in range(rMaxi + 1):
        mask = rad == r
        if np.any(mask):
            e[r] = magnitude[mask].mean()

    return e


def harmonicCount(magnitude: np.ndarray,) -> int:
    mag = removeCenter(magnitude)

    threshold = (np.mean(mag) + 2 * np.std(mag))
    peaks = np.sum(mag > threshold)
    return int(peaks)


def peakreg(magnitude: np.ndarray,) -> float:
    mag = removeCenter(magnitude)
    threshold = (np.mean(mag) + 2 * np.std(mag))
    coords = np.argwhere(mag > threshold)

    if len(coords) < 2:
        return 0.0

    dist = []

    for i in range(len(coords) - 1):
        d = np.linalg.norm(coords[i] - coords[i + 1])
        dist.append(d)
    dist = np.array(dist)

    if len(dist) == 0:
        return 0.0

    return float(1 /(np.std(dist) + 1))


def freqE(gray: np.ndarray,):
    magnitude = computeFFT(gray)
    peak = fftPeakStrength(magnitude)
    mid = midFrequencyRatio(magnitude)
    radial = radialE(magnitude)
    hrmc = harmonicCount(magnitude)
    reg = peakreg(magnitude)

    confidence = np.mean([min(peak / 8, 1), min(mid, 1), min(hrmc / 100, 1), min(reg, 1),])

    return {
        "confidence": float(confidence),
        "features": {
            "fft_peak": float(peak),
            "mid_frequency": float(mid),
            "harmonic_count": int(hrmc),
            "peak_reg": float(reg),
            "radial_energy": radial.tolist(),
        },
    }