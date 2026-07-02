from __future__ import annotations
import numpy as np

def combineConfidence(probability: float, evidence: dict,) -> float: # Combine classifier probability with evidence confidence.
    evidenceConfidence = evidence["confidence"]
    score = (0.7 * probability + 0.3 * evidenceConfidence)
    score = np.clip(score, 0.0, 1.0,)
    return float(score)

def decision(score: float, threshold: float = 0.5,) -> int: # 0 = Real, 1 = Screen.
    return int(score >= threshold)