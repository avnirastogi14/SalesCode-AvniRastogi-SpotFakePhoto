from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

validfiles = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
}


def checkFile(path: str | Path) -> bool:
    # Check if the file has a supported image extension.
    return Path(path).suffix.lower() in validfiles


def loadImage(path: str | Path) -> np.ndarray:
    # Loads image in OpenCV's default BGR format.

    path = Path(path)

    if not checkFile(path):
        raise ValueError(f"Unsupported image format: {path}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    return img


def getSize(path: str | Path) -> tuple[int, int]:
    # Returns image size as (height, width).
    img = loadImage(path)
    return img.shape[:2]


def resizeImage(img: np.ndarray, maxDim: int = 512) -> np.ndarray:
    # Resize while maintaining aspect ratio.

    h, w = img.shape[:2]
    maxi = max(h, w)

    if maxi <= maxDim:
        return img

    scale = maxDim / maxi

    nW = int(w * scale)
    nH = int(h * scale)

    return cv2.resize(
        img,
        (nW, nH),
        interpolation=cv2.INTER_AREA,
    )


def getResizedImage(path: str | Path, maxDim: int = 512) -> np.ndarray:
    # Load + resize image.

    img = loadImage(path)
    img = resizeImage(img, maxDim)

    return img


def getGray(img: np.ndarray) -> np.ndarray:
    # Convert BGR image to grayscale.
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def getRGB(img: np.ndarray) -> np.ndarray:
    # Convert BGR image to RGB.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def saveImage(img: np.ndarray, path: str | Path) -> None:
    # Save image (mostly for debugging).

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(path), img)