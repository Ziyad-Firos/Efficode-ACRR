"""
tests/test_differential.py — behavioural verification correctness.

Why this file exists
--------------------
This is the harness that decides whether a generated refactoring is safe
to show a user. The two tests that matter most are the ones the CodeT5+
plan explicitly demanded: a genuinely correct refactoring must verify as
equivalent, and a DELIBERATELY WRONG one must be rejected. Everything else
here exists because building this module surfaced real bugs along the way
(a parameter type-inference gap, and a speed-measurement design flaw that
made a genuine O(n^2)->O(n) rewrite look like a 0.8x "slowdown") — each of
those is now a permanent regression test, not just a comment.

Like test_sandbox.py, these tests spawn real subprocesses and are slower
than the rest of the suite. That's the cost of testing the real thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.verify.differential import measure_speedup, verify_equivalent  # noqa: E402
from app.verify.inputs import infer_param_kind  # noqa: E402
import ast  # noqa: E402


# ---------------------------------------------------------------------------
# verify_equivalent — the core promise
# ---------------------------------------------------------------------------

def test_correct_refactoring_verifies_as_equivalent():
    """An unambiguous, genuinely behaviour-preserving rewrite (no ordering
    ambiguity from duplicate values -- see the module docstring on why
    find_pair-style 'first match' functions are a bad choice here) must
    pass with zero disagreements."""
    original = (
        "def common_elements(list_a, list_b):\n"
        "    result = []\n"
        "    for x in list_a:\n"
        "        if x in list_b:\n"
        "            result.append(x)\n"
        "    return result\n"
    )
    candidate = (
        "def common_elements(list_a, list_b):\n"
        "    lookup = set(list_b)\n"
        "    result = []\n"
        "    for x in list_a:\n"
        "        if x in lookup:\n"
        "            result.append(x)\n"
        "    return result\n"
    )
    result = verify_equivalent(original, candidate, "common_elements", trials=15)
    assert result.equivalent is True, result.disagreements
    assert result.trials_run > 0
    assert result.trials_agreed == result.trials_run


def test_deliberately_wrong_refactoring_is_rejected():
    """The plan's explicit requirement: a candidate with a real bug (an
    off-by-one in the index stored for later lookup) must be caught, not
    waved through because it parses and looks plausible."""
    original = (
        "def find_pair(nums, target):\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(len(nums)):\n"
        "            if i != j and nums[i] + nums[j] == target:\n"
        "                return (i, j)\n"
        "    return None\n"
    )
    wrong_candidate = (
        "def find_pair(nums, target):\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        complement = target - n\n"
        "        if complement in seen:\n"
        "            return (seen[complement], i)\n"
        "        seen[n] = i + 1\n"  # bug: off by one
        "    return None\n"
    )
    result = verify_equivalent(original, wrong_candidate, "find_pair", trials=15)
    assert result.equivalent is False
    assert len(result.disagreements) > 0


def test_custom_values_override_generic_edge_cases():
    """A naively-recursive function blows up on the generic KIND_INT edge
    case of 10**6 (stack depth / sandbox timeout) -- not a bug in the
    function, just a bad test input for this shape. custom_values must let
    a caller supply small values instead."""
    original = (
        "def fib(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    )
    candidate = (
        "from functools import lru_cache\n\n"
        "@lru_cache(maxsize=None)\n"
        "def fib(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    )
    result = verify_equivalent(
        original, candidate, "fib", trials=10,
        custom_values={"n": [0, 1, 2, 5, 10, 15]},
    )
    assert result.equivalent is True, result.disagreements


def test_unknown_function_name_fails_cleanly():
    result = verify_equivalent("def f(): return 1\n", "def f(): return 1\n", "g", trials=5)
    assert result.equivalent is False
    assert "not found" in result.note


def test_candidate_syntax_error_fails_cleanly():
    result = verify_equivalent("def f(): return 1\n", "def f(: return 1\n", "f", trials=5)
    assert result.equivalent is False
    assert "does not parse" in result.note


def test_no_argument_function_still_verifies():
    result = verify_equivalent(
        "def answer():\n    return 42\n",
        "def answer():\n    return 6 * 7\n",
        "answer",
        trials=5,
    )
    assert result.equivalent is True
    assert result.trials_run == 1  # nothing to vary -- one call is the whole test


# ---------------------------------------------------------------------------
# measure_speedup — regression tests for the two bugs found building it
# ---------------------------------------------------------------------------

def test_real_complexity_change_is_detected():
    """Regression test: this exact case (O(n^2) membership scan -> O(n)
    set lookup) measured as a 0.8x 'slowdown' before two bugs were fixed --
    scaling only one of two collections that both matter, and timing
    subprocess startup instead of the actual computation. Must now show a
    clearly growing ratio and get flagged."""
    original = (
        "def common_elements(list_a, list_b):\n"
        "    result = []\n"
        "    for x in list_a:\n"
        "        if x in list_b:\n"
        "            result.append(x)\n"
        "    return result\n"
    )
    candidate = (
        "def common_elements(list_a, list_b):\n"
        "    lookup = set(list_b)\n"
        "    result = []\n"
        "    for x in list_a:\n"
        "        if x in lookup:\n"
        "            result.append(x)\n"
        "    return result\n"
    )
    result = measure_speedup(original, candidate, "common_elements", sizes=(100, 500, 2000))
    assert result.measured is True
    assert result.complexity_class_likely_changed is True
    ratios = [s.original_seconds / max(s.candidate_seconds, 1e-9) for s in result.samples]
    assert ratios[-1] > ratios[0], "ratio should grow with size for a real complexity-class change"


def test_constant_factor_change_is_not_flagged_as_complexity_change():
    """A loop-to-comprehension rewrite is faster but does NOT change the
    complexity class -- must not be reported as one."""
    original = (
        "def double_all(nums):\n"
        "    result = []\n"
        "    for n in nums:\n"
        "        doubled = n * 2\n"
        "        result.append(doubled)\n"
        "    return result\n"
    )
    candidate = "def double_all(nums):\n    return [n * 2 for n in nums]\n"
    result = measure_speedup(original, candidate, "double_all", sizes=(100, 500, 2000))
    assert result.measured is True
    assert result.complexity_class_likely_changed is False


def test_speedup_refuses_functions_with_no_sized_parameter():
    result = measure_speedup(
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    return a + b\n",
        "add",
    )
    assert result.measured is False
    assert "nothing to scale" in result.note


# ---------------------------------------------------------------------------
# infer_param_kind — regression test for the comparison-operator gap
# ---------------------------------------------------------------------------

def test_infers_int_from_equality_comparison():
    """Regression test: a parameter used only in `x == target` (no
    arithmetic, no subscripting) fell through to KIND_UNKNOWN and got fed
    list/dict/string values in verify_equivalent, which crashed the
    candidate for reasons unrelated to the refactoring under test. Found
    by testing this module against a real two-parameter function."""
    src = (
        "def find_pair(nums, target):\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(len(nums)):\n"
        "            if i != j and nums[i] + nums[j] == target:\n"
        "                return (i, j)\n"
        "    return None\n"
    )
    func_node = ast.parse(src).body[0]
    assert infer_param_kind(func_node, "nums") == "list"
    assert infer_param_kind(func_node, "target") == "int"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
