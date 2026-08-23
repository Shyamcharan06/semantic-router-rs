"""Measures routing accuracy against the labeled dataset.

Usage:
    python eval/run_eval.py [--router-url http://localhost:8080] [--dataset eval/dataset.jsonl]

Hits the router's GET /route debug endpoint (no backend LLM calls involved)
for every labeled prompt, then reports accuracy and a per-category
precision/recall/F1 breakdown plus a confusion matrix. Requires the router
to already be running (see README for how to start it).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from sklearn.metrics import classification_report, confusion_matrix


def load_dataset(path: Path) -> list[dict]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def classify(router_url: str, prompt: str) -> tuple[str, float]:
    resp = requests.get(f"{router_url}/route", params={"q": prompt}, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    return body["category"] or "none", body["score"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", default="http://localhost:8080")
    parser.add_argument("--dataset", default=str(Path(__file__).parent / "dataset.jsonl"))
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    print(f"Loaded {len(dataset)} labeled examples from {args.dataset}")

    try:
        requests.get(f"{args.router_url}/healthz", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"Router not reachable at {args.router_url}: {e}", file=sys.stderr)
        print("Start it first, e.g.: cargo run --release  (see README)", file=sys.stderr)
        sys.exit(1)

    y_true, y_pred, latencies_ms = [], [], []
    for example in dataset:
        start = time.perf_counter()
        predicted, score = classify(args.router_url, example["prompt"])
        latencies_ms.append((time.perf_counter() - start) * 1000)
        y_true.append(example["category"])
        y_pred.append(predicted)

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(dataset)
    avg_latency = sum(latencies_ms) / len(latencies_ms)

    print()
    print(f"Accuracy: {accuracy:.1%} ({correct}/{len(dataset)})")
    print(f"Avg /route latency: {avg_latency:.1f} ms")
    print()
    print("Per-category report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, columns = predicted):")
    header = "".ljust(20) + "".join(label[:12].rjust(14) for label in labels)
    print(header)
    for label, row in zip(labels, matrix):
        print(label.ljust(20) + "".join(str(v).rjust(14) for v in row))

    misses = [(t, p, ex["prompt"]) for t, p, ex in zip(y_true, y_pred, dataset) if t != p]
    if misses:
        print(f"\nMisrouted examples ({len(misses)}):")
        for actual, predicted, prompt in misses:
            print(f"  [{actual} -> {predicted}] {prompt}")


if __name__ == "__main__":
    main()
