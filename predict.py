# from __future__ import annotations
# import sys
# from fusion.classifier import ScreenClassifier
# from fusion.features import extractFeatures, extractEvidence
# from fusion.confidence import combineConfidence
# from utils.image_io import loadImage

# from pathlib import Path

# ROOT = Path(__file__).resolve().parent
# MODEL_PATH = ROOT / "model" / "classifier.pkl"

# def predict(imagePath: str) -> float:
#     img = loadImage(imagePath)
#     features = extractFeatures(img)
#     classifier = ScreenClassifier()
#     classifier.load(MODEL_PATH)
#     evidence = extractEvidence(img)
#     probability = classifier.predict(features)
#     score = combineConfidence(probability, evidence,)

#     return score

# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python predict.py image.jpg")
#         return

#     probability = predict(sys.argv[1])
#     print(f"{probability:.4f}")

# if __name__ == "__main__":
#     main()

from __future__ import annotations
import sys
from fusion.classifier import ScreenClassifier
from fusion.features import extractAll
from fusion.confidence import combineConfidence
from utils.image_io import loadImage

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "classifier.pkl"

def predict(imagePath: str) -> float:
    img = loadImage(imagePath)
    classifier = ScreenClassifier()
    classifier.load(MODEL_PATH)

    features, evidence = extractAll(img)
    probability = classifier.predict(features)
    score = combineConfidence(probability, evidence,)

    return score

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py image.jpg")
        return

    probability = predict(sys.argv[1])
    print(f"{probability:.4f}")

if __name__ == "__main__":
    main()