# from __future__ import annotations
# import numpy as np
# from evidence.frequency import freqE
# from evidence.texture import textureEvidence
# from evidence.optical import opticalEvidence
# from evidence.geometry import geometryEvidence
# from evidence.focus import focusEvidence
# from evidence.compression import compressionEvidence
# from utils.image_io import getGray

# def flattenFeatures(data: dict,) -> list[float]: # Convert feature dictionary into a flat list.
#     values = []
#     for value in data.values():
#         if isinstance(value, list):
#             values.extend(value)

#         else:
#             values.append(value)

#     return values


# def extractFeatures(img: np.ndarray,) -> np.ndarray:
#     gray = getGray(img)
#     frequency = freqE(gray)
#     texture = textureEvidence(gray)
#     optical = opticalEvidence(img)
#     geometry = geometryEvidence(gray)
#     focus = focusEvidence(gray)
#     compression = compressionEvidence(gray)

#     features = []
#     features.extend(flattenFeatures(frequency["features"]))
#     features.extend(flattenFeatures(texture["features"]))
#     features.extend(flattenFeatures(optical["features"]))
#     features.extend(flattenFeatures(geometry["features"]))
#     features.extend(flattenFeatures(focus["features"]))
#     features.extend(flattenFeatures(compression["features"]))

#     return np.array(features, dtype=np.float32,)


# def extractEvidence(img: np.ndarray,):
#     gray = getGray(img)
#     frequency = freqE(gray)
#     texture = textureEvidence(gray)
#     optical = opticalEvidence(img)
#     geometry = geometryEvidence(gray)
#     focus = focusEvidence(gray)
#     compression = compressionEvidence(gray)

#     confidence = np.mean(
#         [
#             frequency["confidence"],
#             texture["confidence"],
#             optical["confidence"],
#             geometry["confidence"],
#             focus["confidence"],
#             compression["confidence"],
#         ]
#     )

#     return {
#         "confidence": float(confidence),
#         "frequency": frequency,
#         "texture": texture,
#         "optical": optical,
#         "geometry": geometry,
#         "focus": focus,
#         "compression": compression,
#     }

# from __future__ import annotations
# import numpy as np
# from evidence.frequency import freqE
# from evidence.texture import textureEvidence
# from evidence.optical import opticalEvidence
# from evidence.geometry import geometryEvidence
# from evidence.focus import focusEvidence
# from evidence.compression import compressionEvidence
# from utils.image_io import getGray

# def flattenFeatures(data: dict,) -> list[float]: # Convert feature dictionary into a flat list.
#     values = []
#     for value in data.values():
#         if isinstance(value, list):
#             values.extend(value)

#         else:
#             values.append(value)

#     return values


# def featureNames() -> list[str]:
#     # Names in the exact order extractFeatures() concatenates them.
#     # Keep this in sync with the "features" dict keys in each evidence module.
#     names = []
#     names.extend(["fft_peak", "mid_frequency", "harmonic_count", "peak_reg",
#                   "radial_energy_mean", "radial_energy_std", "radial_energy_slope",])
#     names.extend(["lbp_uniformity", "entropy", "edge_density",
#                   "local_variance", "gradient_strength",])
#     names.extend(["glare_score", "reflection_gradient",
#                   "saturation_bias", "bright_area_ratio",])
#     names.extend(["line_count", "orientation_std", "hv_ratio", "average_line_length",])
#     names.extend(["laplacian_variance", "tenengrad_score",
#                   "blur_difference", "edge_sharpness",])
#     names.extend(["block_difference", "patch_noise", "ringing_score", "dct_energy",])
#     return names


# def extractFeatures(img: np.ndarray,) -> np.ndarray:
#     gray = getGray(img)
#     frequency = freqE(gray)
#     texture = textureEvidence(gray)
#     optical = opticalEvidence(img)
#     geometry = geometryEvidence(gray)
#     focus = focusEvidence(gray)
#     compression = compressionEvidence(gray)

#     features = []
#     features.extend(flattenFeatures(frequency["features"]))
#     features.extend(flattenFeatures(texture["features"]))
#     features.extend(flattenFeatures(optical["features"]))
#     features.extend(flattenFeatures(geometry["features"]))
#     features.extend(flattenFeatures(focus["features"]))
#     features.extend(flattenFeatures(compression["features"]))

#     return np.array(features, dtype=np.float32,)


# def extractEvidence(img: np.ndarray,):
#     gray = getGray(img)
#     frequency = freqE(gray)
#     texture = textureEvidence(gray)
#     optical = opticalEvidence(img)
#     geometry = geometryEvidence(gray)
#     focus = focusEvidence(gray)
#     compression = compressionEvidence(gray)

#     confidence = np.mean(
#         [
#             frequency["confidence"],
#             texture["confidence"],
#             optical["confidence"],
#             geometry["confidence"],
#             focus["confidence"],
#             compression["confidence"],
#         ]
#     )

#     return {
#         "confidence": float(confidence),
#         "frequency": frequency,
#         "texture": texture,
#         "optical": optical,
#         "geometry": geometry,
#         "focus": focus,
#         "compression": compression,
#     }

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


def featureNames() -> list[str]:
    names = []
    names.extend(["fft_peak", "mid_frequency", "harmonic_count", "peak_reg",
                  "radial_energy_mean", "radial_energy_std", "radial_energy_slope",])
    names.extend(["lbp_uniformity", "entropy", "edge_density",
                  "local_variance", "gradient_strength",])
    names.extend(["glare_score", "reflection_gradient",
                  "saturation_bias", "bright_area_ratio",])
    names.extend(["line_count", "orientation_std", "hv_ratio", "average_line_length",])
    names.extend(["laplacian_variance", "tenengrad_score",
                  "blur_difference", "edge_sharpness",])
    names.extend(["block_difference", "patch_noise", "ringing_score", "dct_energy",])
    return names


def extractAll(img: np.ndarray,):
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

    featureVector = np.array(features, dtype=np.float32,)

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

    evidence = {
        "confidence": float(confidence),
        "frequency": frequency,
        "texture": texture,
        "optical": optical,
        "geometry": geometry,
        "focus": focus,
        "compression": compression,
    }

    return featureVector, evidence


def extractFeatures(img: np.ndarray,) -> np.ndarray:
    features, _ = extractAll(img)
    return features


def extractEvidence(img: np.ndarray,):
    _, evidence = extractAll(img)
    return evidence