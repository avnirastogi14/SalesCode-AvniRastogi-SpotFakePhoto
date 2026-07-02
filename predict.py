from __future__ import annotations
import sys
from fusion.classifier import ScreenClassifier
from fusion.features import extractFeatures
from utils.image_io import loadImage

MODEL_PATH = "model/classifier.pkl"

def predict(imagePath: str) -> float:
    img = loadImage(imagePath)
    features = extractFeatures(img)
    classifier = ScreenClassifier()
    classifier.load(MODEL_PATH)
    probability = classifier.predict(features)
    return float(probability)

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py image.jpg")
        return

    probability = predict(sys.argv[1])
    print(f"{probability:.4f}")

if __name__ == "__main__":
    main()