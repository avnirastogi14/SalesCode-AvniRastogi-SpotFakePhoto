from __future__ import annotations
import cv2
import numpy as np

def showImage(title: str, img: np.ndarray,) -> None:
    cv2.imshow(title, img,)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawEdges(gray: np.ndarray,) -> np.ndarray:
    return cv2.Canny(gray, 100, 200,)

def drawMask(img: np.ndarray, mask: np.ndarray,) -> np.ndarray:
    output = img.copy()
    output[mask > 0] = (0, 0, 255,)
    return output

def normalizeImage(img: np.ndarray,) -> np.ndarray:
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX,).astype(np.uint8)