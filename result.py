# from __future__ import annotations

# import platform
# import statistics
# import sys
# import time

# from fusion.classifier import ScreenClassifier
# from fusion.features import extractFeatures, extractEvidence
# from fusion.confidence import combineConfidence
# from utils.image_io import loadImage

# MODEL_PATH = "model/classifier.pkl"


# def benchmark(imagePath: str, runs: int = 20):

#     classifier = ScreenClassifier()
#     classifier.load(MODEL_PATH)

#     img = loadImage(imagePath)

#     timings = []
#     score = 0.0

#     for _ in range(runs):

#         start = time.perf_counter()

#         features = extractFeatures(img)
#         evidence = extractEvidence(img)

#         probability = classifier.predict(features)

#         score = combineConfidence(
#             probability,
#             evidence,
#         )

#         end = time.perf_counter()

#         timings.append((end - start) * 1000)

#     print()
#     print("Prediction Result")
#     print(f"Score              : {score:.4f}")

#     print()
#     print("Latency")
#     print(f"Average            : {statistics.mean(timings):.2f} ms")
#     print(f"Minimum            : {min(timings):.2f} ms")
#     print(f"Maximum            : {max(timings):.2f} ms")

#     print()
#     print("Device")
#     print(f"System             : {platform.system()}")
#     print(f"Machine            : {platform.machine()}")
#     print(f"Processor          : {platform.processor()}")


# if __name__ == "__main__":

#     if len(sys.argv) != 2:
#         print("Usage: python benchmark.py image.jpg")
#         sys.exit()

#     benchmark(sys.argv[1])

from __future__ import annotations

import platform
import statistics
import sys
import time
from pathlib import Path

from fusion.classifier import ScreenClassifier
from fusion.features import extractAll
from fusion.confidence import combineConfidence
from utils.image_io import loadImage

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "classifier.pkl"


def benchmark(imagePath: str, runs: int = 20):

    classifier = ScreenClassifier()
    classifier.load(MODEL_PATH)

    img = loadImage(imagePath)

    timings = []
    score = 0.0

    for _ in range(runs):

        start = time.perf_counter()

        features, evidence = extractAll(img)
        probability = classifier.predict(features)
        score = combineConfidence(probability, evidence)

        end = time.perf_counter()

        timings.append((end - start) * 1000)

    print()
    print("Prediction Result")
    print(f"Score              : {score:.4f}")

    print()
    print("Latency")
    print(f"Average            : {statistics.mean(timings):.2f} ms")
    print(f"Minimum            : {min(timings):.2f} ms")
    print(f"Maximum            : {max(timings):.2f} ms")

    print()
    print("Device")
    print(f"System             : {platform.system()}")
    print(f"Machine            : {platform.machine()}")
    print(f"Processor          : {platform.processor()}")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python result.py image.jpg")
        sys.exit()

    benchmark(sys.argv[1])