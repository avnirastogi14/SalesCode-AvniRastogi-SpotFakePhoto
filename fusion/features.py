from __future__ import annotations
import numpy as np
from evidence.frequency import freqE
from evidence.texture import textureEvidence
from evidence.optical import opticalEvidence
from evidence.geometry import geometryEvidence
from evidence.focus import focusEvidence
from evidence.compression import compressionEvidence
from utils.image_io import getGray

def flattenFeatures(data: dict,) -> list[float]: # Convert feature dictionary into a flat list.
    values = []
    for value in data.values():
        if isinstance(value, list):
            values.extend(value)

        else:
            values.append(value)

    return values


def extractFeatures(img: np.ndarray,) -> np.ndarray:
    gray = getGray(img)
    frequency = freqE(gray)
    texture = textureEvidence(gray)
    optical = opticalEvidence(img)
    geometry = geometryEvidence(gray)
    focus = focusEvidence(gray)
    compression = compressionEvidence(gray)

    features = []
    features.extend(flattenFeatures(frequency["features"]))
    features.extend(flattenFeatures(texture["features"]))
    features.extend(flattenFeatures(optical["features"]))
    features.extend(flattenFeatures(geometry["features"]))
    features.extend(flattenFeatures(focus["features"]))
    features.extend(flattenFeatures(compression["features"]))

    return np.array(features, dtype=np.float32,)


def extractEvidence(img: np.ndarray,):
    gray = getGray(img)
    frequency = freqE(gray)
    texture = textureEvidence(gray)
    optical = opticalEvidence(img)
    geometry = geometryEvidence(gray)
    focus = focusEvidence(gray)
    compression = compressionEvidence(gray)

    confidence = np.mean(
        [
            frequency["confidence"],
            texture["confidence"],
            optical["confidence"],
            geometry["confidence"],
            focus["confidence"],
            compression["confidence"],
        ]
    )

    return {
        "confidence": float(confidence),
        "frequency": frequency,
        "texture": texture,
        "optical": optical,
        "geometry": geometry,
        "focus": focus,
        "compression": compression,
    }