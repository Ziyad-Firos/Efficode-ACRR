"""
verify_ml_model.py — RandomForest Complexity Predictor Accuracy Test
=====================================================================
Tests the ML model against hand-labeled ground truth across four
difficulty tiers. Uses /refactor endpoint, reads complexity.before.

Run:
    .venv\\Scripts\\python.exe verify_ml_model.py

What each tier is testing:
  Tier 1 — Easy/canonical: standard textbook cases, O(1) through O(2^n).
            If these fail, nothing else matters.
  Tier 2 — Structural traps: cases that fool heuristics counting loop depth.
            Two sequential loops that are NOT O(n^2). One loop with a
            squared bound that IS O(n^2) despite depth=1.
  Tier 3 — Hidden complexity: complexity hiding inside recursion or built-ins,
            zero explicit loops in some cases.
  Tier 4 — Composite/real-world: multi-stage, graph traversal, and the key
            pair: fib_naive vs fib_memo — same AST shape, different complexity
            because of a cache. Same prediction for both = shape-matching not
            reasoning.
"""

import os
import re
import sys
import requests
from dataclasses import dataclass, field
from typing import Optional

BASE = os.environ.get("ACRR_BASE_URL", "http://localhost:8000")

# How each expected label maps to what the model might return
LABEL_VARIANTS: dict[str, list[str]] = {
    "O(1)":        ["o(1)", "constant"],
    "O(log n)":    ["o(log n)", "o(logn)", "logarithmic"],
    "O(n)":        ["o(n)", "linear"],
    "O(n log n)":  ["o(n log n)", "o(nlogn)", "linearithmic", "n log n"],
    "O(n^2)":      ["o(n^2)", "o(n\u00b2)", "o(n**2)", "quadratic"],
    "O(n^3)":      ["o(n^3)", "o(n\u00b3)", "o(n**3)", "cubic"],
    "O(2^n)":      ["o(2^n)", "o(2**n)", "exponential"],
    "O(V+E)":      ["o(v+e)", "o(v + e)", "v+e", "v + e"],
}


@dataclass
class Case:
    tier: int
    name: str
    expected: str
    code: str
    note: str = ""


# ── Test cases ─────────────────────────────────────────────────────────────

CASES: list[Case] = [

    # ── Tier 1: easy / canonical ───────────────────────────────────────────

    Case(1, "constant index access", "O(1)",
         "def get_first(lst):\n    return lst[0]\n"),

    Case(1, "binary search iterative", "O(log n)",
         "def binary_search(arr, target):\n"
         "    lo, hi = 0, len(arr) - 1\n"
         "    while lo <= hi:\n"
         "        mid = (lo + hi) // 2\n"
         "        if arr[mid] == target:\n"
         "            return mid\n"
         "        elif arr[mid] < target:\n"
         "            lo = mid + 1\n"
         "        else:\n"
         "            hi = mid - 1\n"
         "    return -1\n"),

    Case(1, "single pass max", "O(n)",
         "def find_max(lst):\n"
         "    m = lst[0]\n"
         "    for x in lst:\n"
         "        if x > m:\n"
         "            m = x\n"
         "    return m\n"),

    Case(1, "sort call", "O(n log n)",
         "def sort_items(lst):\n"
         "    return sorted(lst)\n"),

    Case(1, "nested loop pair sum", "O(n^2)",
         "def has_pair_sum(nums, target):\n"
         "    for i in range(len(nums)):\n"
         "        for j in range(i + 1, len(nums)):\n"
         "            if nums[i] + nums[j] == target:\n"
         "                return True\n"
         "    return False\n"),

    Case(1, "triple nested loop", "O(n^3)",
         "def has_triple_sum(nums, target):\n"
         "    n = len(nums)\n"
         "    for i in range(n):\n"
         "        for j in range(i + 1, n):\n"
         "            for k in range(j + 1, n):\n"
         "                if nums[i] + nums[j] + nums[k] == target:\n"
         "                    return True\n"
         "    return False\n"),

    Case(1, "naive recursive fibonacci", "O(2^n)",
         "def fib_naive(n):\n"
         "    if n <= 1:\n"
         "        return n\n"
         "    return fib_naive(n - 1) + fib_naive(n - 2)\n"),

    # ── Tier 2: structural traps ───────────────────────────────────────────

    Case(2, "two SEQUENTIAL loops (not nested) -> O(n)", "O(n)",
         "def sum_then_max(lst):\n"
         "    total = 0\n"
         "    for x in lst:\n"
         "        total += x\n"
         "    m = lst[0]\n"
         "    for x in lst:\n"
         "        if x > m:\n"
         "            m = x\n"
         "    return total, m\n",
         note="Heuristic trap: 2 for-loops but they're sequential not nested. Still O(n)."),

    Case(2, "single loop quadratic bound -> O(n^2)", "O(n^2)",
         "def count_pairs(n):\n"
         "    count = 0\n"
         "    for i in range(n * n):\n"
         "        count += 1\n"
         "    return count\n",
         note="Depth=1 but loop bound is n*n. Model relying on depth alone guesses O(n)."),

    Case(2, "triangular nested loop -> O(n^2)", "O(n^2)",
         "def triangular_pairs(n):\n"
         "    count = 0\n"
         "    for i in range(n):\n"
         "        for j in range(i, n):\n"
         "            count += 1\n"
         "    return count\n",
         note="Inner range shrinks each pass — still O(n^2)/2 = O(n^2)."),

    # ── Tier 3: hidden complexity ──────────────────────────────────────────

    Case(3, "'in' on list inside loop -> O(n^2)", "O(n^2)",
         "def find_common(list_a, list_b):\n"
         "    result = []\n"
         "    for x in list_a:\n"
         "        if x in list_b:\n"
         "            result.append(x)\n"
         "    return result\n",
         note="One explicit loop, but 'in' on a list is O(n) — total O(n^2)."),

    Case(3, "recursive factorial no loops -> O(n)", "O(n)",
         "def factorial(n):\n"
         "    if n <= 1:\n"
         "        return 1\n"
         "    return n * factorial(n - 1)\n",
         note="No for/while node. Loop-depth features see this as O(1)."),

    Case(3, "recursive binary search no loops -> O(log n)", "O(log n)",
         "def bsearch(arr, target, lo, hi):\n"
         "    if lo > hi:\n"
         "        return -1\n"
         "    mid = (lo + hi) // 2\n"
         "    if arr[mid] == target:\n"
         "        return mid\n"
         "    elif arr[mid] < target:\n"
         "        return bsearch(arr, target, mid + 1, hi)\n"
         "    else:\n"
         "        return bsearch(arr, target, lo, mid - 1)\n",
         note="No loops. Recursion halves input each time — O(log n)."),

    # ── Tier 4: composite / real-world ─────────────────────────────────────

    Case(4, "sort then scan, dominant term -> O(n log n)", "O(n log n)",
         "def closest_pair(nums):\n"
         "    nums = sorted(nums)\n"
         "    best = float('inf')\n"
         "    for i in range(len(nums) - 1):\n"
         "        best = min(best, nums[i + 1] - nums[i])\n"
         "    return best\n",
         note="sorted() dominates the O(n) scan after it."),

    Case(4, "graph BFS -> O(V+E)", "O(V+E)",
         "from collections import deque\n\n"
         "def bfs(graph, start):\n"
         "    visited = {start}\n"
         "    queue = deque([start])\n"
         "    order = []\n"
         "    while queue:\n"
         "        node = queue.popleft()\n"
         "        order.append(node)\n"
         "        for neighbor in graph[node]:\n"
         "            if neighbor not in visited:\n"
         "                visited.add(neighbor)\n"
         "                queue.append(neighbor)\n"
         "    return order\n",
         note="A different shape of complexity — not just a bigger polynomial."),

    Case(4, "memoized fibonacci -> O(n)  [KEY TEST]", "O(n)",
         "def fib_memo(n, cache={}):\n"
         "    if n in cache:\n"
         "        return cache[n]\n"
         "    if n <= 1:\n"
         "        return n\n"
         "    cache[n] = fib_memo(n - 1, cache) + fib_memo(n - 2, cache)\n"
         "    return cache[n]\n",
         note="CRITICAL: same recursive shape as fib_naive (O(2^n)). "
              "Same prediction for both = shape-matching not reasoning."),
]

AMBIGUOUS: list[Case] = [
    Case(0, "quicksort (avg O(n log n), worst O(n^2))", "O(n log n)",
         "def quicksort(arr):\n"
         "    if len(arr) <= 1:\n"
         "        return arr\n"
         "    pivot = arr[len(arr) // 2]\n"
         "    left  = [x for x in arr if x < pivot]\n"
         "    mid   = [x for x in arr if x == pivot]\n"
         "    right = [x for x in arr if x > pivot]\n"
         "    return quicksort(left) + mid + quicksort(right)\n",
         note="No single correct answer — average O(n log n), worst O(n^2). "
              "Check confidence vs a Tier 1 case: it should be lower here."),
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _matches(expected: str, raw_prediction: str) -> bool:
    """True if raw_prediction contains a known variant of expected label."""
    text = raw_prediction.lower()
    for variant in LABEL_VARIANTS.get(expected, [expected.lower()]):
        if variant in text:
            return True
    return False


def _predict(code: str) -> tuple[Optional[str], float, str]:
    """
    POST to /refactor, return (label, confidence, raw_label).
    Returns (None, 0.0, error_msg) on failure.
    """
    try:
        r = requests.post(
            f"{BASE}/refactor",
            json={"code": code, "use_ai": False, "level": "medium"},
            timeout=15,
        )
    except requests.exceptions.RequestException as e:
        return None, 0.0, f"request error: {e}"

    if r.status_code != 200:
        return None, 0.0, f"HTTP {r.status_code}"

    cx = r.json().get("complexity")
    if cx is None:
        return None, 0.0, "complexity field missing (sklearn not installed?)"

    label = cx.get("before", "")
    confidence = cx.get("confidence", 0.0)
    return label, confidence, label


# ── Runner ─────────────────────────────────────────────────────────────────

def run() -> None:
    print(f"ML Complexity Predictor — Accuracy Test  |  {BASE}\n")

    tier_results: dict[int, list[bool]] = {1: [], 2: [], 3: [], 4: []}
    all_results: list[tuple[Case, bool, str, float]] = []

    for case in CASES:
        label, confidence, raw = _predict(case.code)

        if label is None:
            # Can't determine pass/fail — count as fail, print why
            passed = False
            print(f"  [FAIL] Tier {case.tier} | {case.name}")
            print(f"         error: {raw}\n")
        else:
            passed = _matches(case.expected, raw)
            tag = "PASS" if passed else "FAIL"
            conf_str = f"{confidence*100:.0f}%"
            print(f"  [{tag}] Tier {case.tier} | {case.name}")
            print(f"         expected={case.expected:<12}  got={raw:<14}  conf={conf_str}")
            if case.note:
                print(f"         note: {case.note}")
            print()

        tier_results[case.tier].append(passed)
        all_results.append((case, passed, raw, confidence))

    # ── Key comparison: fib_naive vs fib_memo ─────────────────────────────
    print("=" * 60)
    print("KEY PAIR: fib_naive (O(2^n)) vs fib_memo (O(n))")
    print("Same recursive AST shape — different only by cache dict.")
    print("Same prediction for both = shape-matching, not reasoning.\n")

    fib_naive = next((r for r in all_results if "fib_naive" in r[0].name), None)
    fib_memo  = next((r for r in all_results if "fib_memo"  in r[0].name), None)

    if fib_naive and fib_memo:
        naive_pred = fib_naive[2]
        memo_pred  = fib_memo[2]
        different  = naive_pred.lower() != memo_pred.lower()
        print(f"  fib_naive prediction: {naive_pred}")
        print(f"  fib_memo  prediction: {memo_pred}")
        if different:
            print("  [PASS] Predictions differ — model sees past the shape")
        else:
            print("  [FAIL] Same prediction — model is shape-matching the AST")
    print()

    # ── Ambiguous case ─────────────────────────────────────────────────────
    print("=" * 60)
    print("AMBIGUOUS CASE (no single right answer — check confidence)\n")
    for case in AMBIGUOUS:
        label, confidence, raw = _predict(case.code)
        if label:
            print(f"  {case.name}")
            print(f"  prediction={raw}  conf={confidence*100:.0f}%")
            print(f"  note: {case.note}")

            # Compare confidence to a Tier 1 case (binary search, also O(n log n) adjacent)
            tier1_confs = [r[3] for r in all_results if r[0].tier == 1 and r[1]]
            if tier1_confs:
                avg_t1 = sum(tier1_confs) / len(tier1_confs)
                print(f"  avg Tier 1 confidence: {avg_t1*100:.0f}%  "
                      f"(ambiguous case should be lower)")
        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 60)
    tier_names = {
        1: "Tier 1 — Easy/canonical      ",
        2: "Tier 2 — Structural traps    ",
        3: "Tier 3 — Hidden complexity   ",
        4: "Tier 4 — Composite/real-world",
    }
    total_pass = total_fail = 0
    for tier in sorted(tier_results):
        results = tier_results[tier]
        p = sum(results)
        t = len(results)
        bar = ("█" * p) + ("░" * (t - p))
        total_pass += p
        total_fail += (t - p)
        print(f"  {tier_names[tier]}  {p}/{t}  [{bar}]")

    print(f"\n  Overall: {total_pass}/{total_pass+total_fail} cases matched")
    print()
    print("  Interpretation:")
    print("  - Tier 1 < 6/7 : fundamental accuracy problem, retrain with more data")
    print("  - Tier 2 fails  : model is counting loop depth, not recognising shape")
    print("  - Tier 3 fails  : expected — lightweight model, no semantic understanding")
    print("  - Tier 4 fails  : expected for fib_memo/BFS — real complexity requires semantics")
    print("=" * 60)


if __name__ == "__main__":
    run()
