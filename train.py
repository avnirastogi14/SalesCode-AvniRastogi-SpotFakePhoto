# from __future__ import annotations

# from pathlib import Path
# import numpy as np

# from fusion.features import extractFeatures
# from fusion.classifier import ScreenClassifier

# from utils.image_io import loadImage


# IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",}

# def loadDataset(dataFolder: str = "data",):
#     featureVectors = []
#     labels = []
#     dataFolder = Path(dataFolder)
#     classes = {"real": 0, "screen": 1,}
#     for className, label in classes.items():
#         folder = dataFolder / className

#         if not folder.exists():
#             continue

#         for imagePath in sorted(folder.iterdir()):
#             if imagePath.suffix.lower() not in IMAGE_EXTENSIONS:
#                 continue

#             try:
#                 img = loadImage(str(imagePath))
#                 features = extractFeatures(img)
#                 featureVectors.append(features)
#                 labels.append(label)
#                 print(f"Loaded {imagePath.name}")

#             except Exception as e:
#                 print(f"Skipped {imagePath.name}: {e}")

#     return (np.array(featureVectors, dtype=np.float32), np.array(labels, dtype=np.int32),)


# def trainModel():
#     print("Loading Dataset")
#     X, y = loadDataset()
#     print()
#     print(f"Total Images : {len(y)}")
#     print(f"Real Images  : {np.sum(y == 0)}")
#     print(f"Screen Images: {np.sum(y == 1)}")
#     classifier = ScreenClassifier()
#     print()
#     print("Training Model")
#     classifier.train(X, y)
#     classifier.save("model/classifier.pkl")
#     print()
#     print("Training Complete.")
#     print("Model saved to model/classifier.pkl")


# if __name__ == "__main__":
#     trainModel()

from __future__ import annotations
from pathlib import Path
import numpy as np
from fusion.features import extractFeatures
from fusion.classifier import ScreenClassifier
from utils.image_io import loadImage

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",}

def loadDataset(dataFolder: str = "data",):
    featureVectors = []
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
                img = loadImage(str(imagePath))
                features = extractFeatures(img)
                featureVectors.append(features)
                labels.append(label)

            except Exception as e:
                print(f"Skipped {imagePath.name}: {e}")

    return (np.array(featureVectors, dtype=np.float32), np.array(labels, dtype=np.int32),)

def trainModel():
    print("Loading Dataset")
    X, y = loadDataset()
    print()
    print(f"Total Images : {len(y)}")
    print(f"Real Images  : {np.sum(y == 0)}")
    print(f"Screen Images: {np.sum(y == 1)}")

    classifier = ScreenClassifier()

    print()
    print("Training Model")
    classifier.train(X, y)
    classifier.save("model/classifier.pkl")

    print()
    print("Training Complete.")
    print(f"Training Images : {len(y)}")
    print("Model saved to model/classifier.pkl")

if __name__ == "__main__":
    trainModel()