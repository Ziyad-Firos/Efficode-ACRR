"""
ml/audit.py — Adversarial audit of the complexity model.

evaluate.py asks "how well does it score?". This file asks "are those scores
honest?". It checks the specific ways an ML result can be inflated:

  1. LEAKAGE       Is the "held-out" benchmark actually held out, or are its
                   cases near-duplicates of training samples?
  2. EFFECTIVE N   530 samples sounds like a lot. How many are independent?
  3. SELECTION BIAS Hyperparameters were chosen to maximise grouped CV, then
                   grouped CV was reported. Nested CV removes that circularity.
  4. CALIBRATION   When the model says 95%, is it right 95% of the time?
  5. ROBUSTNESS    Does it hold up on code that looks nothing like the corpus?
  6. LABEL RISK    Which corpus labels are arguable?

Run:  python -m app.ml.audit
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict

from app.ml.benchmark import BENCHMARK
from app.ml.complexity_predictor import CLASS_LABELS, RF_PARAMS, build_dataset
from app.ml.corpus import _BASE, _dedent, _variants, load_corpus
from app.ml.features import extract_features


def header(text: str) -> None:
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 1. Leakage: is the benchmark really held out?
# ---------------------------------------------------------------------------

def check_leakage() -> float:
    header("1. LEAKAGE — is the benchmark actually held out?")

    corpus_vectors = defaultdict(list)
    for code, label in _BASE:
        feats = extract_features(_dedent(code))
        if feats is not None:
            corpus_vectors[tuple(feats)].append(
                (_dedent(code).splitlines()[0], label)
            )

    exact = []
    for name, expected, code in BENCHMARK:
        feats = extract_features(code)
        if feats is None:
            continue
        key = tuple(feats)
        if key in corpus_vectors:
            exact.append((name, corpus_vectors[key][0][0]))

    print(f"  benchmark cases                        : {len(BENCHMARK)}")
    print(f"  with a feature vector identical to a")
    print(f"  corpus algorithm                       : {len(exact)}")
    print()

    if exact:
        print("  These benchmark cases are NOT independent tests. The model")
        print("  sees the exact same 18 numbers during training:")
        for bench_name, corpus_line in exact:
            print(f"    {bench_name:24} == {corpus_line}")
        print()
        print("  A benchmark case with a duplicate feature vector cannot fail")
        print("  unless the model is broken. It measures memorisation, not")
        print("  generalisation.")

    contaminated = len(exact) / len(BENCHMARK)
    print()
    print(f"  >> {contaminated:.0%} of the benchmark is contaminated.")
    if contaminated > 0.3:
        print("  >> The '22/22 held-out benchmark' figure is NOT trustworthy.")
    return contaminated


# ---------------------------------------------------------------------------
# 2. Effective sample size
# ---------------------------------------------------------------------------

def check_effective_n() -> None:
    header("2. EFFECTIVE SAMPLE SIZE — how many samples are independent?")

    sources, labels = load_corpus()
    X, _ = build_dataset()
    distinct = len({tuple(row) for row in X})

    print(f"  reported corpus size                   : {len(sources)}")
    print(f"  independent base algorithms            : {len(_BASE)}")
    print(f"  distinct feature vectors               : {distinct}")
    print()
    print("  The 530 figure counts augmentation. Each base algorithm is")
    print("  expanded into 5 variants that differ only in ways that must not")
    print("  change the label. Those variants teach the model to ignore noise,")
    print("  but they are not new evidence about complexity.")
    print()
    print(f"  >> Quote {len(_BASE)} algorithms, not {len(sources)} samples.")

    per_class = Counter(label for _, label in _BASE)
    print()
    print("  independent algorithms per class:")
    for index, name in enumerate(CLASS_LABELS):
        count = per_class.get(index, 0)
        flag = "  <- thin" if count < 10 else ""
        print(f"    {name:<12} {count:>3}{flag}")


# ---------------------------------------------------------------------------
# 3. Hyperparameter selection bias
# ---------------------------------------------------------------------------

def check_selection_bias() -> None:
    header("3. SELECTION BIAS — nested cross-validation")

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    X, y = build_dataset()
    X = np.array(X)
    y = np.array(y)

    groups = []
    for gid, (code, _) in enumerate(_BASE):
        groups.extend([gid] * len(_variants(code)))
    groups = np.array(groups[: len(y)])

    grid = [
        {"n_estimators": n, "max_depth": d}
        for n in (100, 150, 300)
        for d in (8, 12, None)
    ]

    outer = GroupKFold(n_splits=5)
    outer_scores = []

    for train_idx, test_idx in outer.split(X, y, groups):
        X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # choose hyperparameters using ONLY the training fold
        best_score, best_params = -1.0, grid[0]
        inner = GroupKFold(n_splits=4)
        for params in grid:
            full = {**RF_PARAMS, **params}
            scores = []
            for inner_train, inner_val in inner.split(X_train, y_train, g_train):
                model = RandomForestClassifier(**full)
                model.fit(X_train[inner_train], y_train[inner_train])
                scores.append(model.score(X_train[inner_val], y_train[inner_val]))
            mean = float(np.mean(scores))
            if mean > best_score:
                best_score, best_params = mean, params

        model = RandomForestClassifier(**{**RF_PARAMS, **best_params})
        model.fit(X_train, y_train)
        outer_scores.append(model.score(X_test, y_test))

    nested = float(np.mean(outer_scores))

    plain = RandomForestClassifier(**RF_PARAMS)
    from sklearn.model_selection import cross_val_predict
    predicted = cross_val_predict(plain, X, y, cv=GroupKFold(n_splits=5), groups=groups)
    tuned = float((predicted == y).mean())

    print(f"  grouped CV with the chosen settings    : {tuned * 100:.1f}%")
    print(f"  nested CV (settings chosen per fold)   : {nested * 100:.1f}%")
    print(f"  optimism from tuning on the same data  : {(tuned - nested) * 100:+.1f} pts")
    print()
    print("  The first number was used to PICK n_estimators, then reported as")
    print("  a result. Nested CV re-picks inside each fold, so the test data")
    print("  never influences the choice.")
    print()
    print(f"  >> Report {nested * 100:.1f}%, not {tuned * 100:.1f}%.")


# ---------------------------------------------------------------------------
# 4. Calibration
# ---------------------------------------------------------------------------

def check_calibration() -> None:
    header("4. CALIBRATION — does 95% confidence mean 95% correct?")

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    X, y = build_dataset()
    X = np.array(X)
    y = np.array(y)
    groups = []
    for gid, (code, _) in enumerate(_BASE):
        groups.extend([gid] * len(_variants(code)))
    groups = np.array(groups[: len(y)])

    confidences, correct = [], []
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])
        for row, truth in zip(proba, y[test_idx]):
            index = int(np.argmax(row))
            confidences.append(float(row[index]))
            correct.append(model.classes_[index] == truth)

    confidences = np.array(confidences)
    correct = np.array(correct)

    print(f"  {'confidence band':<20} {'n':>5} {'actual accuracy':>16}  gap")
    print("  " + "-" * 56)
    for low, high in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]:
        mask = (confidences >= low) & (confidences < high)
        if not mask.any():
            continue
        accuracy = correct[mask].mean()
        mean_conf = confidences[mask].mean()
        gap = accuracy - mean_conf
        flag = "  OVERCONFIDENT" if gap < -0.05 else ""
        print(f"  {low:.2f}-{high:.2f}{'':<12} {mask.sum():>5} "
              f"{accuracy * 100:>14.0f}%  {gap * 100:+5.0f} pts{flag}")

    print()
    overall_gap = correct.mean() - confidences.mean()
    print(f"  mean confidence {confidences.mean() * 100:.0f}%  vs  "
          f"actual accuracy {correct.mean() * 100:.0f}%   ({overall_gap * 100:+.0f} pts)")
    if overall_gap < -0.03:
        print("  >> The model is systematically overconfident.")
    elif overall_gap > 0.03:
        print("  >> The model is under-confident (safe direction).")
    else:
        print("  >> Reasonably calibrated.")


# ---------------------------------------------------------------------------
# 5. Robustness on genuinely unfamiliar code
# ---------------------------------------------------------------------------

# Written to be UNLIKE the corpus: odd control flow, library calls, classes
# with state, generators. Every label here is hand-derived, and some are
# deliberately at the edge of what a structural model can see.
STRESS_CASES = [
    ("two pointers converging", "O(n)", '''
def trim_matching_ends(left_items, right_items):
    i = 0
    j = len(right_items) - 1
    matched = 0
    while i < j:
        if left_items[i] == right_items[j]:
            matched = matched + 1
        i = i + 1
        j = j - 1
    return matched
'''),
    ("set intersection", "O(n)", '''
def shared_tags(first, second):
    return set(first).intersection(set(second))
'''),
    ("min() called inside a loop", "O(n\u00b2)", '''
def running_minimums(rows):
    out = []
    for row in rows:
        out.append(min(row))
    return out
'''),
    ("nested while loops", "O(n\u00b2)", '''
def grid_walk(width, height):
    y = 0
    total = 0
    while y < height:
        x = 0
        while x < width:
            total = total + 1
            x = x + 1
        y = y + 1
    return total
'''),
    ("three-branch memoised recursion", "O(n)", '''
def ways(n, cache):
    if n in cache:
        return cache[n]
    if n <= 0:
        return 1
    cache[n] = ways(n - 1, cache) + ways(n - 2, cache) + ways(n - 3, cache)
    return cache[n]
'''),
    ("sorted() over a comprehension", "O(n log n)", '''
def ordered_lengths(words):
    return sorted(len(w) for w in words)
'''),
    ("any() over a list inside a loop", "O(n\u00b2)", '''
def has_common(first, second):
    hits = 0
    for item in first:
        if any(item == other for other in second):
            hits = hits + 1
    return hits
'''),
    ("double enumerate over a matrix", "O(n\u00b2)", '''
def diagonal_total(matrix):
    total = 0
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            if i == j:
                total = total + cell
    return total
'''),
    ("tail recursion with accumulator", "O(n)", '''
def total_from(values, index, running):
    if index >= len(values):
        return running
    return total_from(values, index + 1, running + values[index])
'''),
    ("binary search on the answer", "O(log n)", '''
def smallest_valid(limit):
    low = 1
    high = limit
    while low < high:
        middle = (low + high) // 2
        if middle * middle >= limit:
            high = middle
        else:
            low = middle + 1
    return low
'''),
    ("triple comprehension", "O(n\u00b3+)", '''
def all_combinations(xs, ys, zs):
    return [(a, b, c) for a in xs for b in ys for c in zs]
'''),
    (".count() inside a loop", "O(n\u00b2)", '''
def frequency_report(items):
    report = []
    for item in items:
        report.append(items.count(item))
    return report
'''),
    ("str.join — the correct way", "O(n)", '''
def join_words(words):
    return \'-\'.join(words)
'''),
    ("stack-based traversal", "O(n)", '''
def collect_nodes(root):
    visited = set()
    pending = [root]
    order = []
    while pending:
        node = pending.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            for child in node.children:
                pending.append(child)
    return order
'''),
    ("unmemoised 3-way recursion", "O(2^n)", '''
def paths_to(n):
    if n <= 0:
        return 1
    return paths_to(n - 1) + paths_to(n - 2) + paths_to(n - 3)
'''),
]


def _contaminated_stress_cases() -> set:
    """
    Stress cases whose feature vector also appears in the corpus.

    This set GROWS every time a stress failure is fixed, because fixing it
    means adding that pattern to the training data. That is not cheating —
    it is how the model improves — but it does mean the stress set decays
    into a regression suite and must be refreshed with genuinely new cases
    to stay a generalisation test. Treat a shrinking clean subset as a
    signal to write more cases, not as a rising score.
    """
    corpus_vectors = set()
    for code, _ in _BASE:
        feats = extract_features(_dedent(code))
        if feats is not None:
            corpus_vectors.add(tuple(feats))

    contaminated = set()
    for name, _, code in STRESS_CASES:
        feats = extract_features(code)
        if feats is not None and tuple(feats) in corpus_vectors:
            contaminated.add(name)
    return contaminated


def check_robustness() -> float:
    header("5. ROBUSTNESS — code that looks nothing like the corpus")

    from app.ml.complexity_predictor import predict_label

    contaminated = _contaminated_stress_cases()

    clean_hits = clean_total = 0
    seen_hits = seen_total = 0

    for name, expected, code in STRESS_CASES:
        result = predict_label(code)
        if result is None:
            print("  scikit-learn unavailable — skipped")
            return 0.0
        got, confidence = result
        ok = got == expected
        is_clean = name not in contaminated

        if is_clean:
            clean_total += 1
            clean_hits += ok
        else:
            seen_total += 1
            seen_hits += ok

        flag = "ok  " if ok else "MISS"
        tag = "" if is_clean else "  [seen in training]"
        print(f"  {flag} {name:<32} expect {expected:<11} got {got:<11} "
              f"({confidence:.0%}){tag}")

    print()
    print(f"  unseen patterns   : {clean_hits}/{clean_total}"
          + (f" = {clean_hits / clean_total:.0%}" if clean_total else ""))
    print(f"  regression checks : {seen_hits}/{seen_total}"
          + (f" = {seen_hits / seen_total:.0%}" if seen_total else "")
          + "   (previously-failing patterns now in the corpus)")
    print()
    print("  Only the first line is a generalisation estimate. The second is a")
    print("  regression suite: those patterns were added to training after they")
    print("  failed here, so passing them proves nothing about novel code.")

    accuracy = clean_hits / clean_total if clean_total else 0.0
    if clean_total < 8:
        print()
        print(f"  WARNING: only {clean_total} genuinely unseen cases remain. Add more"
              " before quoting this figure.")
    return accuracy


# ---------------------------------------------------------------------------
# 6. Arguable labels
# ---------------------------------------------------------------------------

def check_labels() -> None:
    header("6. LABEL RISK — corpus labels are hand-assigned")

    arguable = [
        ("quicksort", "labelled O(n log n); worst case is O(n²)"),
        ("bfs / dfs", "labelled O(n); truly O(V+E), which n does not capture"),
        ("merge_sort", "the merge step allocates — space cost is not modelled"),
        ("gcd", "labelled O(log n); Euclid is O(log min(a,b)) in the VALUE, not"
                " the input length"),
        ("count_digits", "same — logarithmic in the value of n, not len(input)"),
        ("fib_memo", "labelled O(n) in n, but arithmetic on big ints is not O(1)"),
        ("list append", "treated as O(1); that is amortised, not worst case"),
    ]
    for name, note in arguable:
        print(f"  {name:<16} {note}")
    print()
    print("  Every label was assigned by hand from textbook complexity. Where")
    print("  the textbook answer is contested, the model learns one opinion.")
    print("  'n' is also not consistently defined: for gcd it is the magnitude")
    print("  of a number, for bfs it is vertices plus edges, for sorting it is")
    print("  element count. The model cannot distinguish these.")


# ---------------------------------------------------------------------------

def main() -> int:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("scikit-learn not installed — cannot audit.")
        return 1

    print("ACRR complexity model — adversarial audit")
    print("Looking for reasons NOT to trust the headline numbers.")

    contamination = check_leakage()
    check_effective_n()
    check_selection_bias()
    check_calibration()
    robustness = check_robustness()
    check_labels()

    header("VERDICT")
    print(f"  benchmark contamination                : {contamination:.0%}")
    print(f"  accuracy on genuinely unseen patterns  : {robustness:.0%}")
    print()
    print("  Use the nested-CV figure and the stress-case figure when")
    print("  describing this model. The benchmark number is a smoke test, not")
    print("  a generalisation estimate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
