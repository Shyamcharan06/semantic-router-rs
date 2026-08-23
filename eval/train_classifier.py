"""Trains a linear probe (multinomial logistic regression) on frozen MiniLM
embeddings, as an alternative to pure cosine-similarity routing.

Trains on exactly the category `examples` already defined in
config/routes.example.yaml -- the same training signal similarity routing
uses -- so the two strategies are a clean apples-to-apples comparison on
eval/dataset.jsonl (which is held out from both).

Gets embeddings from the *running* router's /embed debug endpoint rather
than loading a second copy of the model in Python (e.g. via
sentence-transformers). That keeps exactly one embedding implementation in
the whole project -- the Rust/Candle one -- so there's no risk of the
Python and Rust embeddings silently drifting apart.

Usage:
    python eval/train_classifier.py [--router-url http://localhost:8088]
                                     [--config config/routes.example.yaml]
                                     [--out config/classifier.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import requests
import yaml
from sklearn.linear_model import LogisticRegressionCV


def embed(router_url: str, text: str) -> list[float]:
    resp = requests.get(f"{router_url}/embed", params={"q": text}, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--router-url", default="http://localhost:8088")
    parser.add_argument("--config", default=str(Path(__file__).parent.parent / "config" / "routes.example.yaml"))
    parser.add_argument("--out", default=str(Path(__file__).parent.parent / "config" / "classifier.json"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    try:
        requests.get(f"{args.router_url}/healthz", timeout=5).raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Router not reachable at {args.router_url}: {e}\nStart it first (see README).")

    X, y = [], []
    for category in config["categories"]:
        name = category["name"]
        for example in category["examples"]:
            X.append(embed(args.router_url, example))
            y.append(name)
        print(f"embedded {len(category['examples'])} examples for '{name}'")

    labels = sorted(set(y))
    # Assumes >2 categories: sklearn's binary LogisticRegression only stores
    # one coefficient row (for the positive class), which the class_-index
    # lookup below doesn't handle. Every category set in this repo has 5.
    if len(labels) < 3:
        raise SystemExit("train_classifier.py assumes 3+ categories (binary logistic regression needs different handling)")

    # 60 examples in 384 dimensions is badly underdetermined: an
    # unregularized (or weakly regularized, e.g. sklearn's C=1.0 default)
    # fit separates the training set perfectly but is close to random on
    # anything it hasn't seen (measured: 7% held-out accuracy, worse than
    # guessing). LogisticRegressionCV picks the regularization strength by
    # 5-fold cross-validation *within the training examples* -- never
    # touching eval/dataset.jsonl -- which fixes that.
    clf = LogisticRegressionCV(Cs=np.logspace(-3, 1, 20), cv=5, max_iter=2000)
    clf.fit(X, y)
    print(f"cross-validated C: {clf.C_[0]:.4g}")

    # clf.classes_ is sorted and its order matches coef_'s row order, but we
    # re-index by name into our own `labels` order to make that explicit
    # rather than relying on both sorts agreeing by coincidence.
    label_to_row = {label: i for i, label in enumerate(clf.classes_)}
    weights = [clf.coef_[label_to_row[label]].tolist() for label in labels]
    bias = [float(clf.intercept_[label_to_row[label]]) for label in labels]

    out = {"labels": labels, "weights": weights, "bias": bias}
    Path(args.out).write_text(json.dumps(out, indent=2))

    train_accuracy = clf.score(X, y)
    print(f"\nwrote {args.out}")
    print(f"{len(labels)} labels, {len(X)} training examples, {train_accuracy:.1%} training-set accuracy")
    print("(training accuracy isn't the number that matters -- run eval/run_eval.py with")
    print(" routing.strategy: classifier in routes.yaml for the held-out comparison)")


if __name__ == "__main__":
    main()
