from __future__ import annotations
import cv2
import numpy as np

def normalizeImage(img: np.ndarray)-> np.ndarray: #  Normalize image to range [0,1].
    return img.astype(np.float32) / 255.0

def equalizeHistogram(gray: np.ndarray)-> np.ndarray: # Equalize histogram of grayscale image.
    return cv2.equalizeHist(gray)

def gaussianBlur(img: np.ndarray, kernel: int = 3,) -> np.ndarray: # for edge detection + noise reduction
    return cv2.GaussianBlur(img,(kernel, kernel),0,)


def splitPatches(img: np.ndarray, rows: int = 6, cols: int = 6,)-> list[np.ndarray]: # image --> grids
    h, w = img.shape[:2]

    pH = h // rows
    pW = w // cols

    patches = []

    for i in range(rows):
        for j in range(cols):

            y1 = i * pH
            y2 = (i + 1) * pH

            x1 = j * pW
            x2 = (j + 1) * pW

            patches.append(img[y1:y2, x1:x2])
    return patches


def imageEntropy(gray: np.ndarray)-> float:
    hist = cv2.calcHist([gray],[0],None,[256],[0, 256],)
    hist = hist.ravel()
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)

def imageContrast(gray: np.ndarray)-> float: # Standard deviation of intensity.
    return float(np.std(gray))


def cropBorder(img: np.ndarray, percent: float = 0.05,) -> np.ndarray: # Crop a small border around image.
    h, w = img.shape[:2]
    dh = int(h*percent)
    dw = int(w*percent)
    return img[dh:h - dh,dw:w - dw,]