# from __future__ import annotations
# from pathlib import Path
# import numpy as np
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,)
# from fusion.classifier import ScreenClassifier
# from fusion.features import extractFeatures
# from utils.image_io import loadImage

# IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",}

# def loadDataset(dataFolder: str = "data",):
#     images = []
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
#                 img = loadImage(imagePath)
#                 images.append(extractFeatures(img))
#                 labels.append(label)

#             except Exception as e:
#                 print(f"Skipped {imagePath.name}: {e}")

#     return (np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32),)


# def evaluate():
#     classifier = ScreenClassifier()
#     classifier.load("model/classifier.pkl")
#     X, y = loadDataset()
#     predictions = []
#     probabilities = []
#     for feature in X:
#         probability = classifier.predict(feature)
#         probabilities.append(probability)
#         predictions.append(int(probability >= 0.5))
#     predictions = np.array(predictions)
#     print()
#     print("Evaluation Results\n")
#     print(f"Accuracy : {accuracy_score(y, predictions):.4f}")
#     print(f"Precision: {precision_score(y, predictions):.4f}")
#     print(f"Recall   : {recall_score(y, predictions):.4f}")
#     print(f"F1 Score : {f1_score(y, predictions):.4f}")
#     print()

#     print("Confusion Matrix")
#     print(confusion_matrix(y, predictions,))


# if __name__ == "__main__":
#     evaluate()


# from __future__ import annotations

# from pathlib import Path

# import numpy as np

# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,)
# from fusion.classifier import ScreenClassifier
# from fusion.features import extractFeatures
# from utils.image_io import loadImage

# IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",}

# def loadDataset(dataFolder: str = "data"):
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

#             except Exception as e:
#                 print(f"Skipped {imagePath.name}: {e}")

#     return (np.array(featureVectors, dtype=np.float32), np.array(labels, dtype=np.int32),)


# def evaluate():
#     X, y = loadDataset()
#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42,)

#     accuracies = []
#     precisions = []
#     recalls = []
#     f1scores = []

#     fold = 1

#     for trainIndex, testIndex in skf.split(X, y):
#         Xtrain = X[trainIndex]
#         Xtest = X[testIndex]
#         ytrain = y[trainIndex]
#         ytest = y[testIndex]
#         classifier = ScreenClassifier()
#         classifier.train(Xtrain, ytrain)
#         predictions = []
#         for feature in Xtest:
#             probability = classifier.predict(feature)
#             predictions.append(int(probability >= 0.5))

#         predictions = np.array(predictions)

#         accuracy = accuracy_score(ytest, predictions)
#         precision = precision_score(ytest, predictions)
#         recall = recall_score(ytest, predictions)
#         f1 = f1_score(ytest, predictions)

#         accuracies.append(accuracy)
#         precisions.append(precision)
#         recalls.append(recall)
#         f1scores.append(f1)

#         print()
#         print(f"Fold {fold}")
#         print(f"Accuracy : {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall   : {recall:.4f}")
#         print(f"F1 Score : {f1:.4f}")
#         fold += 1

#     print()
#     print("=" * 35)
#     print("Average over 5 folds")
#     print("=" * 35)

#     print(f"Accuracy : {np.mean(accuracies):.4f}")
#     print(f"Precision: {np.mean(precisions):.4f}")
#     print(f"Recall   : {np.mean(recalls):.4f}")
#     print(f"F1 Score : {np.mean(f1scores):.4f}")


# if __name__ == "__main__":
#     evaluate()

from __future__ import annotations
 
from pathlib import Path
 
import numpy as np
 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,)
from fusion.classifier import ScreenClassifier
from fusion.features import extractFeatures, featureNames
from utils.image_io import loadImage
 
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",}
 
def loadDataset(dataFolder: str = "data"):
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
 
 
def evaluate():
    X, y = loadDataset()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42,)
 
    accuracies = []
    precisions = []
    recalls = []
    f1scores = []
 
    fold = 1
 
    for trainIndex, testIndex in skf.split(X, y):
        Xtrain = X[trainIndex]
        Xtest = X[testIndex]
        ytrain = y[trainIndex]
        ytest = y[testIndex]
        classifier = ScreenClassifier()
        classifier.train(Xtrain, ytrain)
        predictions = []
        for feature in Xtest:
            probability = classifier.predict(feature)
            predictions.append(int(probability >= 0.5))
 
        predictions = np.array(predictions)
 
        accuracy = accuracy_score(ytest, predictions)
        precision = precision_score(ytest, predictions)
        recall = recall_score(ytest, predictions)
        f1 = f1_score(ytest, predictions)
 
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1scores.append(f1)
 
        print()
        print(f"Fold {fold}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        fold += 1
 
    print()
    print("=" * 35)
    print("Average over 5 folds")
    print("=" * 35)
 
    print(f"Accuracy : {np.mean(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall   : {np.mean(recalls):.4f}")
    print(f"F1 Score : {np.mean(f1scores):.4f}")
 
    print()
    print("=" * 35)
    print("Feature Importances (full-data fit)")
    print("=" * 35)
 
    finalClassifier = ScreenClassifier()
    finalClassifier.train(X, y)
 
    importances = finalClassifier.model.feature_importances_
    names = featureNames()
 
    ranked = sorted(zip(names, importances), key=lambda pair: pair[1], reverse=True)
 
    for name, importance in ranked:
        print(f"{name:<22}: {importance:.4f}")
 
 
if __name__ == "__main__":
    evaluate()