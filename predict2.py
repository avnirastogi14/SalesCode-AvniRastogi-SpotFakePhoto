from __future__ import annotations

import sys

from fusion.classifier2 import ScreenClassifier
from fusion.features import extractFeatures, extractEvidence
from fusion.confidence import combineConfidence
from utils.image_io import loadImage


MODEL_PATH = "model/classifier2.pkl"


def predict(imagePath: str) -> float:

    img = loadImage(imagePath)

    features = extractFeatures(img)

    classifier = ScreenClassifier()

    classifier.load(MODEL_PATH)

    evidence = extractEvidence(img)

    probability = classifier.predict(features)

    score = combineConfidence(
        probability,
        evidence,
    )

    return score


def main():

    if len(sys.argv) != 2:

        print("Usage: python predict2.py image.jpg")
        return

    probability = predict(sys.argv[1])

    print(f"{probability:.4f}")


if __name__ == "__main__":

    main()