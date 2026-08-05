"""
ml/complexity_predictor.py — RandomForest Big-O complexity predictor.

Architecture
------------
    corpus.py    real labeled Python functions (source, not vectors)
        │
    features.py  18-feature structural extractor
        │
    this file    RandomForestClassifier + explanation layer

The model is still a RandomForest, as before. What changed is where its
training data comes from: features are now extracted from real parsed Python
source by the same extractor used at prediction time, instead of being
hand-typed as rows of numbers. See corpus.py for why that matters.

Complexity classes
------------------
  0 O(1)   1 O(log n)   2 O(n)   3 O(n log n)   4 O(n²)   5 O(n³+)   6 O(2^n)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from app.models import ComplexityPrediction
from app.ml.features import extract_features, FEATURE_COUNT, FEATURE_NAMES

logger = logging.getLogger("acrr.ml.complexity_predictor")

_MODEL_PATH = Path(__file__).parent / "models" / "complexity_rf.pkl"
_META_PATH = Path(__file__).parent / "models" / "complexity_rf.meta"

CLASS_LABELS = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(n³+)", "O(2^n)"]

# Hyperparameters live here so the eval harness trains an identical model.
#
# n_estimators=150 was chosen by measurement, not by habit. Swept against
# grouped cross-validation on the 530-sample corpus:
#     300 trees -> 97.0%   2.9 MB
#     200 trees -> 97.0%   2.0 MB
#     150 trees -> 98.1%   1.5 MB   <- best accuracy AND half the size
#     100 trees -> 97.9%   1.0 MB
#      60 trees -> 96.6%   0.6 MB
# More trees was not better here: past ~150 the extra trees mostly re-fit the
# same structural splits. The smaller model also matters for deployment,
# where a free-tier instance has a hard memory ceiling.
RF_PARAMS = dict(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced",
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_dataset() -> Tuple[List[List[int]], List[int]]:
    """
    Extract features for every corpus sample.

    Returns (X, y). Samples that fail to parse are dropped with a warning
    rather than silently — a corpus entry that does not parse is a bug in the
    corpus, and should be visible.
    """
    from app.ml.corpus import load_corpus

    sources, labels = load_corpus()
    X: List[List[int]] = []
    y: List[int] = []

    for source, label in zip(sources, labels):
        feats = extract_features(source)
        if feats is None:
            logger.warning("Corpus sample failed to parse — skipped: %r", source[:60])
            continue
        X.append(feats)
        y.append(label)

    return X, y


def _corpus_fingerprint() -> str:
    """
    Hash of the corpus + feature definition.

    A cached .pkl trained on an older corpus or an older feature set is worse
    than no cache at all, because it fails silently. Storing this alongside the
    model lets _load_model() detect staleness and retrain.
    """
    from app.ml import corpus as corpus_mod
    from app.ml import features as features_mod

    h = hashlib.sha256()
    for mod in (corpus_mod, features_mod):
        try:
            h.update(Path(mod.__file__).read_bytes())
        except Exception:  # noqa: BLE001
            pass
    return h.hexdigest()[:16]


def train_model(save: bool = True):
    """Train the RandomForest on the corpus. Returns the fitted model or None."""
    try:
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        import joblib  # type: ignore
    except ImportError:
        logger.warning(
            "scikit-learn/joblib not installed — complexity prediction disabled. "
            "Install with: pip install scikit-learn joblib"
        )
        return None

    X, y = build_dataset()
    if not X:
        logger.error("Training corpus is empty — cannot train")
        return None

    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X, y)

    if save:
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, _MODEL_PATH)
        _META_PATH.write_text(_corpus_fingerprint(), encoding="utf-8")
        logger.info("Trained on %d samples — saved to %s", len(X), _MODEL_PATH)

    return clf


def _load_model():
    """Load the cached model, retraining if it is missing, stale, or mismatched."""
    if _MODEL_PATH.exists():
        try:
            import joblib  # type: ignore

            saved_fp = _META_PATH.read_text(encoding="utf-8").strip() if _META_PATH.exists() else ""
            if saved_fp != _corpus_fingerprint():
                logger.info("Corpus or features changed since the model was cached — retraining")
                raise ValueError("stale cache")

            clf = joblib.load(_MODEL_PATH)
            if getattr(clf, "n_features_in_", FEATURE_COUNT) != FEATURE_COUNT:
                logger.warning(
                    "Cached model expects %d features, extractor produces %d — retraining",
                    clf.n_features_in_, FEATURE_COUNT,
                )
                raise ValueError("feature count mismatch")

            logger.info("Complexity predictor loaded from %s", _MODEL_PATH)
            return clf
        except Exception as exc:  # noqa: BLE001
            logger.info("Rebuilding complexity model: %s", exc)

    return train_model()


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model


# ---------------------------------------------------------------------------
# Explanation layer
# ---------------------------------------------------------------------------

def explain(features: List[int]) -> List[str]:
    """
    Turn a feature vector into plain-language structural evidence.

    This does not attempt to reconstruct the forest's decision path. It reports
    the structures the extractor actually found, so a user can check the
    reasoning against their own code instead of being handed a bare label.
    """
    f = dict(zip(FEATURE_NAMES, features))
    notes: List[str] = []

    depth = f["max_loop_depth"]
    if depth >= 3:
        notes.append(f"{depth} levels of nested loops — work grows like n^{depth}")
    elif depth == 2:
        notes.append("two nested loops — each element is compared against every other")
    elif depth == 1:
        if f["loop_count"] > 1:
            notes.append(
                f"{f['loop_count']} loops, but none nested — sequential passes stay linear"
            )
        else:
            notes.append("a single pass over the input")
    elif not f["has_recursion"]:
        notes.append("no loops and no recursion — the work does not grow with input size")

    if f["comprehension_count"]:
        notes.append(
            f"{f['comprehension_count']} comprehension(s) counted as loops "
            "(they iterate just like a for-loop)"
        )

    if f["loop_bound_is_quadratic"]:
        notes.append("a loop bound built from a multiplication, e.g. range(n * n)")

    if f["max_self_calls_per_path"] >= 2:
        if f["has_memo_guard"]:
            notes.append(
                "branching recursion, but results are cached — each input is solved once"
            )
        elif f["recursion_halves_input"]:
            notes.append(
                "divide and conquer — two recursive calls, each on roughly half the input"
            )
        else:
            notes.append(
                f"{f['max_self_calls_per_path']} recursive calls per invocation with no "
                "caching — the call tree doubles at every level"
            )
    elif f["max_self_calls_per_path"] == 1:
        if f["recursion_halves_input"] or f["while_loop_halves"]:
            notes.append("each step discards half the remaining input")
        else:
            notes.append("one recursive call per invocation, shrinking the input by a constant")
    elif f["while_loop_halves"]:
        notes.append("a while-loop that halves its range each iteration")

    if f["has_sorting_call"]:
        notes.append("a sort, which costs n log n on its own")

    if f["linear_scan_in_loop"]:
        notes.append(
            "a linear scan inside a loop (`in` against a list, or .index()/sum()) — "
            "this is a hidden quadratic"
        )

    if f["has_visited_guard"]:
        notes.append(
            "a set-backed visited guard — nested in shape, but each item is processed once"
        )

    return notes


def suggest_improvement(features: List[int]) -> Optional[str]:
    """The single highest-value structural change, or None if nothing obvious."""
    f = dict(zip(FEATURE_NAMES, features))

    if f["linear_scan_in_loop"]:
        return (
            "Convert the list being searched into a set (or a dict) before the loop. "
            "Membership testing drops from O(n) to O(1), taking the whole function "
            "from quadratic to linear."
        )
    if f["max_self_calls_per_path"] >= 2 and not f["has_memo_guard"] \
            and not f["recursion_halves_input"]:
        return (
            "Add memoisation — cache results by argument and return the cached value "
            "on re-entry. This collapses an exponential call tree to linear."
        )
    if f["max_loop_depth"] >= 2 and not f["has_visited_guard"]:
        return (
            "Consider whether the inner loop can be replaced by a hash map lookup. "
            "Many nested-loop problems (pair sums, duplicate finding) become a single "
            "pass with a dict."
        )
    if f["loop_bound_is_quadratic"]:
        return (
            "The loop bound is a product, so this runs n² times despite looking like a "
            "single loop. Check whether the full product range is really needed."
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_complexity(
    original_code: str,
    refactored_code: str,
) -> Optional[ComplexityPrediction]:
    """
    Predict Big-O for the original and refactored code.

    Returns None when scikit-learn is unavailable or the code does not parse,
    so the caller degrades to a response without a complexity section rather
    than failing.
    """
    try:
        model = _get_model()
        if model is None:
            return None

        feat_before = extract_features(original_code)
        feat_after = extract_features(refactored_code)
        if feat_before is None or feat_after is None:
            return None

        import numpy as np  # type: ignore

        proba_before = model.predict_proba([feat_before])[0]
        proba_after = model.predict_proba([feat_after])[0]

        classes = list(getattr(model, "classes_", range(len(CLASS_LABELS))))
        label_before = CLASS_LABELS[classes[int(np.argmax(proba_before))]]
        label_after = CLASS_LABELS[classes[int(np.argmax(proba_after))]]
        confidence = round(float(max(proba_before)), 2)

        return ComplexityPrediction(
            before=label_before,
            after=label_after,
            confidence=confidence,
            explanation=explain(feat_before),
            suggestion=suggest_improvement(feat_before),
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Complexity prediction failed: %s", exc)
        return None


def predict_label(code: str) -> Optional[Tuple[str, float]]:
    """Convenience helper used by the eval harness: returns (label, confidence)."""
    model = _get_model()
    if model is None:
        return None
    feats = extract_features(code)
    if feats is None:
        return None

    import numpy as np  # type: ignore

    proba = model.predict_proba([feats])[0]
    classes = list(getattr(model, "classes_", range(len(CLASS_LABELS))))
    return CLASS_LABELS[classes[int(np.argmax(proba))]], round(float(max(proba)), 2)
