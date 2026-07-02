from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def computeCost(latencyMs: float, pricePerHour: float, cores: int = 1) -> dict:
    imagesPerHour = (3600 / (latencyMs / 1000)) * cores
    costPerImage = pricePerHour / imagesPerHour

    return {
        "latency_ms": latencyMs,
        "price_per_hour": pricePerHour,
        "cores": cores,
        "images_per_hour": imagesPerHour,
        "cost_per_image": costPerImage,
        "cost_per_1k": costPerImage * 1_000,
        "cost_per_1m": costPerImage * 1_000_000,
    }


def printReport(result: dict) -> None:
    print()
    print("Cost Estimate")
    print("=" * 40)
    print(f"Latency            : {result['latency_ms']:.2f} ms/image")
    print(f"Instance price     : ${result['price_per_hour']:.4f}/hr "
          f"({result['cores']} core(s), single-threaded assumed)")
    print(f"Throughput         : {result['images_per_hour']:.0f} images/hour")
    print()
    print(f"Cost per image     : ${result['cost_per_image']:.8f}")
    print(f"Cost per 1,000     : ${result['cost_per_1k']:.4f}")
    print(f"Cost per 1,000,000 : ${result['cost_per_1m']:.2f}")
    print()
    print("Note: compute time only. Excludes request/network overhead,")
    print("cold starts, and orchestration cost, which can push real-world")
    print("cost up several-fold. On-device inference is $0 (runs on the")
    print("user's phone, no server needed).")


def main():
    parser = argparse.ArgumentParser(description="Estimate cloud cost per image.")
    parser.add_argument("--latency_ms", type=float, default=None,
                         help="Average latency per image in ms (e.g. from result.py)")
    parser.add_argument("--image", type=str, default=None,
                         help="Image path to benchmark live instead of supplying --latency_ms")
    parser.add_argument("--runs", type=int, default=20,
                         help="Number of runs when benchmarking live (default: 20)")
    parser.add_argument("--price_per_hour", type=float, default=0.017,
                         help="Cloud instance price in $/hour (default: 0.017, small ARM CPU)")
    parser.add_argument("--cores", type=int, default=1,
                         help="Number of parallel cores/workers assumed (default: 1)")
    args = parser.parse_args()

    if args.latency_ms is None and args.image is None:
        parser.error("Provide either --latency_ms or --image")

    if args.latency_ms is not None:
        latencyMs = args.latency_ms
    else:
        import statistics
        import time
        from fusion.classifier import ScreenClassifier
        from fusion.features import extractAll
        from fusion.confidence import combineConfidence
        from utils.image_io import loadImage

        classifier = ScreenClassifier()
        classifier.load(ROOT / "model" / "classifier.pkl")
        img = loadImage(args.image)

        timings = []
        for _ in range(args.runs):
            start = time.perf_counter()
            features, evidence = extractAll(img)
            probability = classifier.predict(features)
            combineConfidence(probability, evidence)
            timings.append((time.perf_counter() - start) * 1000)

        latencyMs = statistics.mean(timings)
        print(f"\nMeasured latency: {latencyMs:.2f} ms average over {args.runs} runs")

    result = computeCost(latencyMs, args.price_per_hour, args.cores)
    printReport(result)


if __name__ == "__main__":
    main()