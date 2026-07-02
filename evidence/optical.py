from __future__ import annotations
import cv2
import numpy as np

def detectBrightRegions(gray: np.ndarray, threshold: int = 235,) -> np.ndarray:
    # Detects over-exposed regions that could correspond to screen glare.
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY,)
    return mask

def glareClusterScore(mask: np.ndarray,) -> float: # Scores whether bright pixels form one compact glare region.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num <= 1:
        return 0.0

    largest = stats[1:, cv2.CC_STAT_AREA].max()
    total = np.count_nonzero(mask)

    if total == 0:
        return 0.0

    return float(largest / total)

def reflectionGradientScore(gray: np.ndarray, mask: np.ndarray,) -> float: # Reflection regions usually have smoother gradients.

    gradX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gradY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad = np.sqrt(gradX**2 + gradY**2)

    values = grad[mask > 0]

    if len(values) == 0:
        return 0.0

    return float(values.mean())

def saturationBias(img: np.ndarray, mask: np.ndarray,) -> float: # Compare saturation inside glare vs outside glare.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    inside = sat[mask > 0]
    outside = sat[mask == 0]

    if len(inside) == 0 or len(outside) == 0:
        return 0.0

    return float(abs(inside.mean() - outside.mean()))

def brightAreaRatio(mask: np.ndarray,) -> float: # Percentage of bright pixels.
    return float(np.count_nonzero(mask)/mask.size)

def opticalEvidence(img: np.ndarray,):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY,)
    mask = detectBrightRegions(gray)
    glare = glareClusterScore(mask)
    reflection = reflectionGradientScore(gray, mask,)

    saturation = saturationBias(img, mask,)

    bright = brightAreaRatio(mask)
    confidence = np.mean([min(glare, 1), min(bright * 8, 1), min(saturation / 60, 1),])

    return {
        "confidence": float(confidence),
        "features": {
            "glare_score": float(glare),
            "reflection_gradient": float(reflection),
            "saturation_bias": float(saturation),
            "bright_area_ratio": float(bright),
        },
    }