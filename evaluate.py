from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,)
from fusion.classifier import ScreenClassifier
from fusion.features import extractFeatures
from utils.image_io import loadImage

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",}

def loadDataset(dataFolder: str = "data",):
    images = []
    labels = []
    dataFolder = Path(dataFolder)
    classes = {"real": 0, "screen": 1,}
    for className, label in classes.items():
        folder = dataFolder / className
        if not folder.exists():
            continue

        for imagePath in sorted(folder.iterdir()):
            if imagePath.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                img = loadImage(imagePath)
                images.append(extractFeatures(img))
                labels.append(label)

            except Exception as e:
                print(f"Skipped {imagePath.name}: {e}")

    return (np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32),)


def evaluate():
    classifier = ScreenClassifier()
    classifier.load("model/classifier.pkl")
    X, y = loadDataset()
    predictions = []
    probabilities = []
    for feature in X:
        probability = classifier.predict(feature)
        probabilities.append(probability)
        predictions.append(int(probability >= 0.5))
    predictions = np.array(predictions)
    print()
    print("Evaluation Results\n")
    print(f"Accuracy : {accuracy_score(y, predictions):.4f}")
    print(f"Precision: {precision_score(y, predictions):.4f}")
    print(f"Recall   : {recall_score(y, predictions):.4f}")
    print(f"F1 Score : {f1_score(y, predictions):.4f}")
    print()

    print("Confusion Matrix")
    print(confusion_matrix(y, predictions,))


if __name__ == "__main__":
    evaluate()