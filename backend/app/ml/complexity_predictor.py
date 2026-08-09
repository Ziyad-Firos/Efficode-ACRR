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

import ast
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from app.models import ComplexityPrediction, FunctionComplexity
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
        notes.append("one loop nested inside another — work can grow quadratically with input size")
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
    """
    The single highest-value structural change, or None if nothing obvious.

    There used to be a fourth branch here: "any nested loop without a
    visited-guard" suggested a hash-map rewrite, unconditionally. That fires
    on genuine lookup/duplicate-finding code, but just as readily on in-place
    sorting, matrix operations, or Gaussian elimination — cases where there
    is no list being searched and the advice is simply wrong. Confirmed on a
    real sample: a nested swap-based sort got told to convert something to a
    dict. `linear_scan_in_loop` below is the actual evidence for a hash-map
    fix; without it, silence is more honest than a guess.
    """
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
    if f["loop_bound_is_quadratic"]:
        return (
            "The loop bound is a product, so this runs n² times despite looking like a "
            "single loop. Check whether the full product range is really needed."
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _predict_one(model, code_or_tree) -> Optional[Tuple[str, float, List[int]]]:
    """Predict a single unit of code. Returns (label, confidence, features)."""
    import numpy as np  # type: ignore

    if isinstance(code_or_tree, str):
        feats = extract_features(code_or_tree)
    else:
        from app.ml.features import extract_features_from_tree
        feats = extract_features_from_tree(code_or_tree)

    if feats is None:
        return None

    proba = model.predict_proba([feats])[0]
    classes = list(getattr(model, "classes_", range(len(CLASS_LABELS))))
    index = classes[int(np.argmax(proba))]
    return CLASS_LABELS[index], round(float(max(proba)), 2), feats


def _top_level_functions(code: str) -> List[ast.AST]:
    """
    Top-level functions and methods, in source order.

    Nested helper functions are left inside their parent — they are part of
    that function's cost, not separate units.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    found: List[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(node)
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found.append(member)
    return found


def analyse_functions(code: str, model) -> List[FunctionComplexity]:
    """
    Predict Big-O for each function separately.

    Analysing a whole file as one unit is what produced the low-confidence
    averages: a file holding one O(n²) function and one O(n) function has
    the loop structure of both flattened together, and the model reports
    something in between at ~52% confidence. Predicted separately those same
    two functions come back at 99% and 92%.
    """
    results: List[FunctionComplexity] = []

    for node in _top_level_functions(code):
        # Wrap the function in its own module so features are scoped to it
        module = ast.Module(body=[node], type_ignores=[])
        prediction = _predict_one(model, module)
        if prediction is None:
            continue
        label, confidence, feats = prediction
        results.append(FunctionComplexity(
            name=node.name,
            line=getattr(node, "lineno", None),
            complexity=label,
            confidence=confidence,
            explanation=explain(feats),
            suggestion=suggest_improvement(feats),
        ))

    # The slowest function determines how the whole file scales.
    if results:
        rank = {label: i for i, label in enumerate(CLASS_LABELS)}
        dominant = max(results, key=lambda f: rank.get(f.complexity, 0))
        dominant.is_dominant = True

    return results


def predict_complexity(
    original_code: str,
    refactored_code: str,
) -> Optional[ComplexityPrediction]:
    """
    Predict Big-O for the original and refactored code.

    Each function is analysed on its own; the slowest one drives the headline
    figure, because that is what actually governs how the file scales. Code
    with no function definitions falls back to whole-module analysis.

    Returns None when scikit-learn is unavailable or the code does not parse,
    so the caller degrades to a response without a complexity section rather
    than failing.
    """
    try:
        model = _get_model()
        if model is None:
            return None

        functions = analyse_functions(original_code, model)

        if functions:
            dominant = next(f for f in functions if f.is_dominant)
            label_before = dominant.complexity
            confidence = dominant.confidence
            explanation = list(dominant.explanation)
            suggestion = dominant.suggestion

            if len(functions) > 1:
                explanation.insert(
                    0,
                    f"'{dominant.name}' is the slowest of {len(functions)} functions "
                    f"and sets the overall complexity.",
                )

            after = analyse_functions(refactored_code, model)
            label_after = (
                next(f.complexity for f in after if f.is_dominant)
                if after else label_before
            )
        else:
            # No function definitions — analyse the module as a whole.
            before = _predict_one(model, original_code)
            after = _predict_one(model, refactored_code)
            if before is None or after is None:
                return None
            label_before, confidence, feats = before
            label_after = after[0]
            explanation = explain(feats)
            suggestion = suggest_improvement(feats)

        return ComplexityPrediction(
            before=label_before,
            after=label_after,
            confidence=confidence,
            explanation=explanation,
            suggestion=suggestion,
            functions=functions,
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
