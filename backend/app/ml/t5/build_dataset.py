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
import random
import sys
from pathlib import Path

from app.ml.t5.pairs import FAMILIES, generate_pairs
from app.verify.differential import measure_speedup, verify_equivalent

logger = logging.getLogger("acrr.ml.t5.build_dataset")

_OUT_PATH = Path(__file__).parent / "pairs_validated.jsonl"
_HOLDOUT_FRACTION = 0.2
_VERIFY_TRIALS = 12
_VERIFY_TIMEOUT = 3.0


def _holdout_families(seed: int = 7) -> set:
    names = [f.name for f in FAMILIES]
    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)
    n_holdout = max(1, round(len(names) * _HOLDOUT_FRACTION))
    return set(shuffled[:n_holdout])


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
