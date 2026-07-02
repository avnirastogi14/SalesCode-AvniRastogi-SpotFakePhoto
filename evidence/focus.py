from __future__ import annotations
import cv2
import numpy as np
from utils.preprocessing import gaussianBlur

def laplacianVariance(gray: np.ndarray,) -> float: # Variance of Laplacian measures image sharpness.
    lap = cv2.Laplacian(gray, cv2.CV_64F,)
    return float(lap.var())

def tenengradScore(gray: np.ndarray,) -> float: # Gradient based focus measure.
    gx = cv2.Sobel( gray, cv2.CV_64F, 1, 0,)
    gy = cv2.Sobel( gray, cv2.CV_64F, 0, 1,)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(grad))

def blurDifference(gray: np.ndarray,) -> float: # Difference between original and blurred image.
    blur = gaussianBlur(gray, kernel=5,)
    diff = cv2.absdiff(gray, blur,)

    return float(np.mean(diff))

def edgeSharpness(gray: np.ndarray,) -> float:
    edges = cv2.Canny(gray, 100, 200,)
    values = gray[edges > 0]
    if len(values) == 0:
        return 0.0
    
    return float(np.std(values))

def focusEvidence(gray: np.ndarray,):
    lap = laplacianVariance(gray)
    ten = tenengradScore(gray)
    blur = blurDifference(gray)
    sharp = edgeSharpness(gray)
    confidence = np.mean([min(lap / 400, 1), min(ten / 60, 1), min(blur / 20, 1), min(sharp / 60, 1),])

    return {
        "confidence": float(confidence),
        "features": {
            "laplacian_variance": float(lap),
            "tenengrad_score": float(ten),
            "blur_difference": float(blur),
            "edge_sharpness": float(sharp),
        },
    }