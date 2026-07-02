from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from utils.preprocessing import ( imageEntropy, splitPatches,)

def computeLBP(gray: np.ndarray, radius: int = 2, points: int = 16,) -> np.ndarray: # Compute Local Binary Pattern image.
    lbp = local_binary_pattern(gray,points,radius,method="uniform",)
    return lbp

def lbpUniformity(lbp: np.ndarray,) -> float:
    hist, _ = np.histogram(lbp.ravel(),bins=np.arange(20),density=True,)
    return float(hist.max())

def edgeDensity(gray: np.ndarray,) -> float:
    edges = cv2.Canny(gray, 100, 200)

    return float(
        np.count_nonzero(edges)
        / edges.size
    )

def localVariance(gray: np.ndarray,) -> float:
    patches = splitPatches(gray, rows=6, cols=6,)
    variances = []
    for patch in patches:
        variances.append(np.var(patch))

    return float(np.mean(variances))

def gradientStrength(gray: np.ndarray,) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1,)
    grad = np.sqrt(gx**2 + gy**2)

    return float(grad.mean())

def textureEvidence(gray: np.ndarray,):
    lbp = computeLBP(gray)
    uniformity = lbpUniformity(lbp)
    entropy = imageEntropy(gray)
    edge = edgeDensity(gray)
    variance = localVariance(gray)
    gradient = gradientStrength(gray)
    confidence = np.mean([uniformity, min(edge * 5, 1), min(gradient / 50, 1),])

    return {
        "confidence": float(confidence),
        "features": {
            "lbp_uniformity": float(uniformity),
            "entropy": float(entropy),
            "edge_density": float(edge),
            "local_variance": float(variance),
            "gradient_strength": float(gradient),
        },
    }

