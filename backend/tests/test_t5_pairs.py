"""
tests/test_t5_pairs.py — training-pair template correctness.

Why this file exists
--------------------
Building app/ml/t5/pairs.py found two real template bugs (index_to_dict's
naive dict-comprehension silently keeping the LAST occurrence of a
duplicate key instead of the first, matching .index()'s semantics) and two
real gaps in the shared type-inference heuristic (app/verify/inputs.py) it
depends on: sorted(param) used in a plain assignment cast no vote at all,
and .pop() being dict-exclusive outvoted a genuine list(param) signal on
the same parameter. All four are permanent regressions here, not just
comments in pairs.py.

This runs the FULL validation pipeline (verify_equivalent + measure_speedup)
against every family, at a SMALL instance count (2 per family, not
build_dataset.py's default 25) specifically to keep this in the regular
test suite's time budget while still exercising every template for real —
build_dataset.py's own run (python -m app.ml.t5.build_dataset) is the full
250-pair validated dataset (10 families x 25), run separately, not as part
of pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ast  # noqa: E402

from app.ml.t5.build_dataset import _holdout_families, validate_and_build  # noqa: E402
from app.ml.t5.pairs import FAMILIES, generate_pairs  # noqa: E402
from app.verify.differential import verify_equivalent  # noqa: E402


def test_every_family_produces_parseable_before_and_after():
    """Cheap, fast sanity check with no subprocess execution — every
    template's output must at least be valid Python defining the claimed
    function name, before spending any time on real verification."""
    pairs = generate_pairs(instances_per_family=2, seed=1)
    assert len(pairs) == len(FAMILIES) * 2
    for pair in pairs:
        before_tree = ast.parse(pair.before)
        after_tree = ast.parse(pair.after)
        before_names = {n.name for n in ast.walk(before_tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        after_names = {n.name for n in ast.walk(after_tree)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert pair.function_name in before_names
        assert pair.function_name in after_names


def test_every_family_passes_differential_verification():
    """The load-bearing test: every template's before/after pair must be
    behaviourally equivalent. A failure here means a template has a real
    bug (see this file's docstring for two that were caught exactly this
    way) — never weaken this assertion to make a broken template pass."""
    pairs = generate_pairs(instances_per_family=2, seed=2)
    failures = []
    for pair in pairs:
        result = verify_equivalent(
            pair.before, pair.after, pair.function_name,
            trials=8, timeout=3.0,
            custom_values=pair.custom_values or None,
        )
        if not result.equivalent:
            failures.append((pair.family, result.note, result.disagreements))
    assert failures == [], f"{len(failures)} template(s) failed equivalence: {failures}"


def test_holdout_is_by_family_not_by_row():
    holdout = _holdout_families()
    all_families = {f.name for f in FAMILIES}
    assert holdout < all_families  # proper subset, not empty, not everything
    # Pinned explicitly (see build_dataset.py's _HOLDOUT_FAMILY_NAMES) so
    # results stay comparable across training runs as more trained
    # families get added -- this is no longer a fraction-of-however-many-
    # families-exist calculation, it's these two, specifically, always.
    assert holdout == {"minmax_in_loop_to_running", "pop0_to_deque"}


def test_build_dataset_reports_family_membership_correctly():
    """Every pair generated for a held-out family must be labelled
    'holdout', every other pair 'train' — no row-level leakage between
    the two splits."""
    result = validate_and_build(instances_per_family=2, write=False)
    assert result["failed"] == 0
    holdout = _holdout_families()
    for row in result["pairs"]:
        expected_split = "holdout" if row["family"] in holdout else "train"
        assert row["split"] == expected_split


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
