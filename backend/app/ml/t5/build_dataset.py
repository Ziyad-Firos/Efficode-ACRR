"""
ml/t5/build_dataset.py — validate every training pair, then save it.

Run as: python -m app.ml.t5.build_dataset

Every (before, after) pair from pairs.py is checked before being trusted:
  1. verify_equivalent() — do they compute the same thing?
  2. measure_speedup() — does the "after" side's speed trend look like a
     real complexity-class change, or just noise?

A pair that fails equivalence is a bug in its TEMPLATE (see pairs.py's
module docstring for why one shape was already rejected this way before
generation even started). This script reports failures loudly and REFUSES
to write a dataset file while any exist, rather than silently dropping bad
rows — "a dataset that has passed differential testing on every row" is
the whole point; one with even a single silently-skipped failure isn't that.

Hold-out is by FAMILY, not by row. Testing on one family's 25th instance
after training on its other 24 measures nothing — it's the same
transformation with different variable names, which a model may have
simply memorised the shape of rather than learned. Holding out entire
families and reporting how many of THOSE still show the expected
complexity-class change is the only way to see whether this approach even
has a chance of generalising, before any fine-tuning is attempted.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from app.ml.t5.pairs import FAMILIES, generate_pairs
from app.verify.differential import measure_speedup, verify_equivalent

logger = logging.getLogger("acrr.ml.t5.build_dataset")

_OUT_PATH = Path(__file__).parent / "pairs_validated.jsonl"
_VERIFY_TRIALS = 12
_VERIFY_TIMEOUT = 3.0

# Pinned explicitly, not derived by shuffling whatever families happen to
# be in FAMILIES at the moment. This used to be a seeded shuffle over
# FAMILIES taking the first 20% -- which meant every time a family got
# added (or removed), the shuffle's result over the new, differently-sized
# list could pick DIFFERENT families as held out, silently invalidating
# every prior training run's "34% pass rate, generalizes to 2/2 held-out
# families" result, since those numbers are only meaningful when compared
# against the same held-out set. Pinning it here means new trained
# families (see pairs.py's running_sum_to_cumulative / running_count_to_
# incremental, added specifically to help the model with these two) can
# be added freely without silently moving the goalposts.
_HOLDOUT_FAMILY_NAMES = frozenset({"minmax_in_loop_to_running", "pop0_to_deque"})


def _holdout_families() -> set:
    """
    Two integrity checks, both real `raise`s rather than `assert` --
    `assert` is silently stripped entirely under `python -O` /
    PYTHONOPTIMIZE=1, which would defeat the exact "fail loudly instead of
    silently invalidating the comparison" purpose this pinning exists for
    in the first place (caught in review, not assumed safe just because
    the happy path was tested).
    """
    all_names = {f.name for f in FAMILIES}

    missing = _HOLDOUT_FAMILY_NAMES - all_names
    if missing:
        raise ValueError(
            f"Pinned held-out family name(s) not found in FAMILIES: {missing} -- "
            "a family was renamed or removed in pairs.py without updating "
            "_HOLDOUT_FAMILY_NAMES here."
        )

    # The old fraction-based formula (max(1, round(len(names) * 0.2))) could
    # never reach 100% of FAMILIES for any size >= 2 -- pinning an explicit
    # set removed that structural guarantee, so it's re-added explicitly:
    # if every non-held-out family were ever removed from pairs.py, this
    # would otherwise silently return the entire FAMILIES set, labelling
    # every generated pair "holdout" and none "train".
    if len(all_names) <= len(_HOLDOUT_FAMILY_NAMES):
        raise ValueError(
            f"_HOLDOUT_FAMILY_NAMES ({sorted(_HOLDOUT_FAMILY_NAMES)}) would cover all of "
            f"FAMILIES ({sorted(all_names)}) -- nothing would be left to train on."
        )

    return set(_HOLDOUT_FAMILY_NAMES)


def validate_and_build(instances_per_family: int = 25, write: bool = True) -> dict:
    pairs = generate_pairs(instances_per_family=instances_per_family)
    holdout = _holdout_families()

    print(f"Generated {len(pairs)} pairs across {len(FAMILIES)} families.")
    print(f"Held-out families ({len(holdout)}): {sorted(holdout)}")
    print()

    validated = []
    failures = []

    for i, pair in enumerate(pairs):
        eq = verify_equivalent(
            pair.before, pair.after, pair.function_name,
            trials=_VERIFY_TRIALS, timeout=_VERIFY_TIMEOUT,
            custom_values=pair.custom_values or None,
        )
        if not eq.equivalent:
            first = eq.disagreements[0] if eq.disagreements else None
            detail = eq.note or (
                f"disagreed on {first.call_expr}: original={first.original_outcome}, "
                f"candidate={first.candidate_outcome}" if first else "no detail"
            )
            failures.append((pair, "equivalence", detail))
            continue

        speed = measure_speedup(pair.before, pair.after, pair.function_name)

        validated.append({
            "family": pair.family,
            "complexity_change": pair.complexity_change,
            "before": pair.before,
            "after": pair.after,
            "function_name": pair.function_name,
            "custom_values": pair.custom_values,  # e.g. naive_fib_to_memo's small
            # int values -- without this, a consumer re-verifying a saved row
            # (see finetune_codet5p.ipynb's gate-checking cell) would fall back
            # to generic edge cases and could blow the sandbox timeout/recursion
            # limit on a family that needs constrained values. Not currently
            # triggered (neither held-out family needs custom_values today) but
            # a real, latent gap if that ever changes -- fixed rather than left
            # as a footnote.
            "split": "holdout" if pair.family in holdout else "train",
            "verified_trials": f"{eq.trials_agreed}/{eq.trials_run}",
            "speedup_measured": speed.measured,
            "complexity_class_likely_changed": speed.complexity_class_likely_changed,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)} checked...")

    print()
    print(f"Validated: {len(validated)}/{len(pairs)}")

    if failures:
        print(f"FAILED: {len(failures)} pairs — these are template bugs, not dropped silently:")
        for pair, kind, detail in failures[:10]:
            print(f"  [{pair.family}] {kind}: {detail}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")
        print()
        print("Refusing to write a dataset file while failures exist.")
        print("Fix the template(s) above — do not delete this check.")
        if write:
            sys.exit(1)
        return {"validated": len(validated), "failed": len(failures), "pairs": []}

    by_family: dict = {}
    for row in validated:
        by_family.setdefault(row["family"], []).append(row)

    print()
    print("Per-family complexity-class-change detection:")
    for family in FAMILIES:
        rows = by_family.get(family.name, [])
        changed = sum(1 for r in rows if r["complexity_class_likely_changed"])
        split = "HOLDOUT" if family.name in holdout else "train"
        print(
            f"  [{split:7}] {family.name:30} {changed}/{len(rows)} showed a growing "
            f"speedup trend  ({family.complexity_change})"
        )

    if write:
        _OUT_PATH.write_text(
            "\n".join(json.dumps(row) for row in validated),
            encoding="utf-8",
        )
        print()
        print(f"Wrote {len(validated)} validated pairs to {_OUT_PATH}")

    return {"validated": len(validated), "failed": len(failures), "pairs": validated}


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    validate_and_build()
