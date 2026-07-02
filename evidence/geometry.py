from __future__ import annotations
import cv2
import numpy as np
from utils.preprocessing import gaussianBlur

def unpackLine(line):
    line = np.asarray(line).reshape(-1)
    return line[0], line[1], line[2], line[3]

def detectEdges(gray: np.ndarray,) -> np.ndarray: # Detect strong image edges.
    gray = gaussianBlur(gray, kernel=3)
    edges = cv2.Canny(gray, 80, 160,)
    return edges


def detectLines(edges: np.ndarray,): # Detect straight lines using Hough Transform.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10,)
    return lines

def lineCount(lines,) -> int:
    if lines is None:
        return 0

    return int(len(lines))


def dominantOrientation(lines,) -> float:
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = unpackLine(line)
        angle = np.degrees(np.arctan2(y2 - y1,x2 - x1,))
        angles.append(angle)
    angles = np.array(angles)

    return float(np.std(angles))


def horizontalVerticalRatio(lines,) -> float:
    if lines is None:
        return 0.0

    hv = 0
    for line in lines:
        x1, y1, x2, y2 = unpackLine(line)
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1,)))

        if angle < 15 or angle > 75:
            hv += 1

    return float(hv / len(lines))


def averageLineLength(lines,) -> float:
    if lines is None:
        return 0.0
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = unpackLine(line)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        lengths.append(length)
    return float(np.mean(lengths))

def geometryEvidence(gray: np.ndarray,):
    edges = detectEdges(gray)
    lines = detectLines(edges)
    count = lineCount(lines)
    orientation = dominantOrientation(lines)
    ratio = horizontalVerticalRatio(lines)
    avgLength = averageLineLength(lines)
    confidence = np.mean([min(count / 80, 1), ratio, min(avgLength / 150, 1),])

    return {
        "confidence": float(confidence),
        "features": {
            "line_count": int(count),
            "orientation_std": float(orientation),
            "hv_ratio": float(ratio),
            "average_line_length": float(avgLength),
        },
    }