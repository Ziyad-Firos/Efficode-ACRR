"""
ml/evaluate.py — Honest evaluation of the complexity predictor.

Run with:
    python -m app.ml.evaluate

Reports three things, in increasing order of how much they should be trusted:

  1. Cross-validated accuracy on the training corpus.
     Stratified 5-fold. Tells you whether the features separate the classes
     at all. Optimistic, because variants of the same base algorithm can land
     in both the train and test folds.

  2. Grouped cross-validation.
     Same folds, but every variant of a base algorithm is forced into the SAME
     fold. This is the number that matters: it measures whether the model
     generalises to an algorithm it has never seen, not whether it can
     recognise a renamed copy of one it has.

  3. Held-out benchmark.
     22 DSA functions written independently of the corpus. The final check.

Anything that reports only (1) is reporting the flattering number.
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import List

from app.ml.complexity_predictor import CLASS_LABELS, RF_PARAMS, build_dataset
from app.ml.corpus import load_corpus, class_distribution
from app.ml.features import extract_features, FEATURE_NAMES
from app.ml.benchmark import BENCHMARK


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        print("scikit-learn is not installed — cannot evaluate.")
        print("  pip install scikit-learn joblib numpy")
        return False


def _confusion(y_true: List[int], y_pred: List[int]) -> None:
    n = len(CLASS_LABELS)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    width = max(len(l) for l in CLASS_LABELS) + 1
    header = " " * (width + 2) + "".join(f"{i:>6}" for i in range(n))
    print(header)
    for i, row in enumerate(matrix):
        cells = "".join(f"{v:>6}" if v else "     ." for v in row)
        total = sum(row)
        correct = row[i]
        rate = f"{correct / total * 100:5.0f}%" if total else "    -"
        print(f"{CLASS_LABELS[i]:>{width}}  {cells}   {rate}")
    print(f"{'':>{width}}  (rows = actual, columns = predicted)")


def main() -> int:
    if not _require_sklearn():
        return 1

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict

    X, y = build_dataset()
    X_arr = np.array(X)
    y_arr = np.array(y)

    print("=" * 72)
    print("TRAINING CORPUS")
    print("=" * 72)
    sources, _ = load_corpus()
    print(f"  {len(sources)} samples, {len(FEATURE_NAMES)} features")
    for name, count in class_distribution().items():
        bar = "█" * (count // 3)
        print(f"    {name:<12} {count:>4}  {bar}")

    # ── 1. Stratified CV (optimistic) ──────────────────────────────────────
    print()
    print("=" * 72)
    print("1. STRATIFIED 5-FOLD CV  (optimistic — variants leak across folds)")
    print("=" * 72)
    clf = RandomForestClassifier(**RF_PARAMS)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(clf, X_arr, y_arr, cv=skf)
    acc = float((pred == y_arr).mean())
    print(f"  accuracy: {acc * 100:.1f}%")

    # ── 2. Grouped CV (honest) ─────────────────────────────────────────────
    # Every variant of a base algorithm shares a group id, so a base algorithm
    # is never split across train and test.
    print()
    print("=" * 72)
    print("2. GROUPED 5-FOLD CV  (honest — unseen algorithms only)")
    print("=" * 72)
    from app.ml.corpus import _BASE, _variants

    groups: List[int] = []
    for gid, (code, _label) in enumerate(_BASE):
        groups.extend([gid] * len(_variants(code)))
    groups_arr = np.array(groups[: len(y_arr)])

    gkf = GroupKFold(n_splits=5)
    grouped_pred = cross_val_predict(
        RandomForestClassifier(**RF_PARAMS), X_arr, y_arr, cv=gkf, groups=groups_arr
    )
    grouped_acc = float((grouped_pred == y_arr).mean())
    print(f"  accuracy: {grouped_acc * 100:.1f}%")
    print()
    _confusion(list(y_arr), list(grouped_pred))

    # ── Feature importances ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("FEATURE IMPORTANCE")
    print("=" * 72)
    fitted = RandomForestClassifier(**RF_PARAMS).fit(X_arr, y_arr)
    ranked = sorted(
        zip(FEATURE_NAMES, fitted.feature_importances_), key=lambda kv: -kv[1]
    )
    for name, importance in ranked:
        bar = "▇" * int(importance * 100)
        print(f"  {name:<26} {importance:6.3f}  {bar}")

    noisy = {"branch_count", "has_subscript", "total_statements"}
    noise_share = sum(i for n, i in ranked if n in noisy)
    print()
    print(f"  share taken by known-irrelevant features: {noise_share * 100:.1f}%")
    print("  (corpus variants deliberately vary these — lower is better)")

    # ── 3. Held-out benchmark ──────────────────────────────────────────────
    print()
    print("=" * 72)
    print("3. HELD-OUT BENCHMARK  (written independently of the corpus)")
    print("=" * 72)
    hits = 0
    failures = []
    for name, expected, code in BENCHMARK:
        feats = extract_features(code)
        proba = fitted.predict_proba([feats])[0]
        predicted = CLASS_LABELS[fitted.classes_[int(np.argmax(proba))]]
        conf = float(max(proba))
        ok = predicted == expected
        hits += ok
        flag = "ok " if ok else "MISS"
        print(f"  {flag}  {name:<26} expected {expected:<11} got {predicted:<11} ({conf:.0%})")
        if not ok:
            failures.append((name, expected, predicted))

    print()
    print(f"  benchmark accuracy: {hits}/{len(BENCHMARK)} = {hits / len(BENCHMARK) * 100:.0f}%")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  stratified CV (optimistic) : {acc * 100:.1f}%")
    print(f"  grouped CV (honest)        : {grouped_acc * 100:.1f}%")
    print(f"  held-out benchmark         : {hits / len(BENCHMARK) * 100:.0f}%")
    if failures:
        print()
        print("  remaining benchmark failures:")
        for name, exp, got in failures:
            print(f"    {name}: expected {exp}, got {got}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
