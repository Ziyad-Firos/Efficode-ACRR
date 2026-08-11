"""
verify/differential.py — proving a refactoring didn't change behaviour.

You cannot write unit tests for code you have never seen. A user pastes a
bare function into a textarea — no test file, no callers, no type hints
guaranteeing intent. But you CAN compare two programs against each other,
and there are always two: the original and the candidate refactoring.

Neither program is "correct" in the abstract. But if the refactoring is
faithful, they must agree on every input. One disagreement is proof of a
bug. Agreement across many inputs is strong evidence, not proof — every
result below is reported that way, never as a guarantee.

This is the same principle as this project's most valuable existing test,
test_refactor_preserves_behaviour (tests/test_refactor_rules.py) — it
executes code and compares results instead of inspecting structure, which
is why it catches real bugs structural checks miss. This module is that
same idea, generalised to arbitrary submitted functions instead of a fixed
set of known test cases.
"""

from __future__ import annotations

import ast
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.verify.inputs import (
    KIND_LIST,
    KIND_STRING,
    build_call_expr,
    build_test_cases,
    generate_edge_case_values,
    generate_sized_value,
    infer_param_kind,
)
from app.verify.sandbox import SandboxResult, run_isolated

DEFAULT_TRIALS = 20
DEFAULT_TIMEOUT = 3.0
DEFAULT_SPEED_SIZES: Tuple[int, ...] = (100, 500, 2000)
DEFAULT_SPEED_TIMEOUT = 5.0


@dataclass
class Disagreement:
    call_expr: str
    original_outcome: str
    candidate_outcome: str


@dataclass
class VerificationResult:
    equivalent: bool
    trials_run: int
    trials_agreed: int
    disagreements: List[Disagreement] = field(default_factory=list)
    note: str = ""


@dataclass
class SizeSample:
    size: int
    original_seconds: float
    candidate_seconds: float


@dataclass
class SpeedupResult:
    measured: bool
    samples: List[SizeSample] = field(default_factory=list)
    complexity_class_likely_changed: bool = False
    note: str = ""


def _find_function(tree: ast.AST, func_name: str) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return node
    return None


def _outcome_signature(result: SandboxResult) -> str:
    """
    A comparable summary of one sandbox run: either its returned value, or
    just the TYPE of exception it raised. Exception message text can
    legitimately differ between equivalent implementations (different
    wording of the same ValueError) without that being a real behavioural
    difference — comparing only the type avoids flagging that as a bug.
    """
    if result.timed_out:
        return "__TIMEOUT__"
    if result.memory_exceeded:
        return "__MEMORY_EXCEEDED__"
    if not result.ok:
        return "__ERROR__:" + (result.error or "").split(":")[0]
    return repr(result.return_value)


def verify_equivalent(
    original: str,
    candidate: str,
    func_name: str,
    trials: int = DEFAULT_TRIALS,
    timeout: float = DEFAULT_TIMEOUT,
    seed: int = 1234,
) -> VerificationResult:
    """
    Run both versions on the same generated inputs; compare results.

    Deterministic (fixed seed) — reproducible for a report, not a fresh
    random sample every call. Runs the original and candidate for each
    trial concurrently (thread pool — these are I/O-bound subprocess
    waits, not CPU-bound work, so no GIL contention) to roughly halve wall
    time versus running every sandbox call in sequence.
    """
    try:
        orig_tree = ast.parse(original)
    except SyntaxError as exc:
        return VerificationResult(False, 0, 0, note=f"original does not parse: {exc}")

    func_node = _find_function(orig_tree, func_name)
    if func_node is None:
        return VerificationResult(False, 0, 0, note=f"function '{func_name}' not found in original code")

    try:
        ast.parse(candidate)
    except SyntaxError as exc:
        return VerificationResult(False, 0, 0, note=f"candidate does not parse: {exc}")

    param_order = [a.arg for a in func_node.args.args]
    rng = random.Random(seed)

    if not param_order:
        test_cases: List[list] = [[]]
    else:
        kinds = {p: infer_param_kind(func_node, p) for p in param_order}
        inputs_per_param = {p: generate_edge_case_values(kinds[p], rng) for p in param_order}
        test_cases = build_test_cases(inputs_per_param, param_order, trials)

    if not test_cases:
        return VerificationResult(False, 0, 0, note="no test cases generated")

    def _run_one(case: list) -> Tuple[str, str, str]:
        call_expr = build_call_expr(func_name, case)
        with ThreadPoolExecutor(max_workers=2) as pool:
            orig_future = pool.submit(run_isolated, original, call_expr, timeout)
            cand_future = pool.submit(run_isolated, candidate, call_expr, timeout)
            orig_result = orig_future.result()
            cand_result = cand_future.result()
        return call_expr, _outcome_signature(orig_result), _outcome_signature(cand_result)

    agreed = 0
    disagreements: List[Disagreement] = []
    for case in test_cases:
        call_expr, orig_sig, cand_sig = _run_one(case)
        if orig_sig == cand_sig:
            agreed += 1
        else:
            disagreements.append(Disagreement(
                call_expr=call_expr,
                original_outcome=orig_sig,
                candidate_outcome=cand_sig,
            ))

    return VerificationResult(
        equivalent=len(disagreements) == 0,
        trials_run=len(test_cases),
        trials_agreed=agreed,
        disagreements=disagreements[:5],  # cap what gets carried around/reported
    )


def _build_timed_call_expr(func_name: str, args: List, repeats: int) -> str:
    """
    A single expression that calls func_name(*args) `repeats` times and
    brackets the loop with two timestamps, all inside the sandboxed
    process — NOT timed from outside via wall-clock around run_isolated().

    Why this matters: subprocess + interpreter startup on this machine
    costs roughly 70ms per run_isolated() call (see sandbox.py's module
    docstring on the child-of-child interpreter behaviour found while
    building it). An O(n) vs O(n^2) difference at n=2000 is real but can
    be single-digit milliseconds in absolute terms — completely buried
    under 70ms of fixed overhead if timed from outside. Timing repeated
    calls INSIDE one process amortises that fixed cost away and measures
    the actual computation, confirmed empirically: without this, a real
    set-based O(n) rewrite of an O(n^2) membership scan measured as a
    1.0x "speedup" (pure noise) at every tested size.

    The result comes back through the same JSON channel run_isolated()
    already uses: [t_start, result_1, ..., result_N, t_end]. The parent
    reads the two floats off the ends and computes (t_end - t_start) /
    repeats -- no changes needed to sandbox.py itself.
    """
    args_repr = ", ".join(repr(a) for a in args)
    return (
        f"[__import__('time').perf_counter()] + "
        f"[{func_name}({args_repr}) for _ in range({repeats})] + "
        f"[__import__('time').perf_counter()]"
    )


def measure_speedup(
    original: str,
    candidate: str,
    func_name: str,
    sizes: Tuple[int, ...] = DEFAULT_SPEED_SIZES,
    timeout: float = DEFAULT_SPEED_TIMEOUT,
    repeats: int = 30,
) -> SpeedupResult:
    """
    Time both at several input sizes. Report the ratio AND how it trends.

    A refactoring that is 1.1x faster at n=100 and 1.2x faster at n=2000
    just shaved a constant factor. One that is 1.1x at n=100 and 40x at
    n=2000 changed the complexity class — only the second is worth calling
    an optimisation. That distinction is exactly what
    complexity_class_likely_changed reports, instead of a single ratio
    that can't tell the two apart.

    Only measures speed of code already known to behave the same —
    callers should run verify_equivalent first and only call this on
    candidates that passed. A refactoring that's fast because it's wrong
    is not an optimisation.
    """
    try:
        orig_tree = ast.parse(original)
    except SyntaxError as exc:
        return SpeedupResult(False, note=f"original does not parse: {exc}")

    func_node = _find_function(orig_tree, func_name)
    if func_node is None:
        return SpeedupResult(False, note=f"function '{func_name}' not found")

    param_order = [a.arg for a in func_node.args.args]
    if not param_order:
        return SpeedupResult(False, note="function takes no arguments -- nothing to scale")

    kinds = {p: infer_param_kind(func_node, p) for p in param_order}
    scalable = [p for p in param_order if kinds[p] in (KIND_LIST, KIND_STRING)]
    if not scalable:
        return SpeedupResult(False, note="no parameter looks sized (list/string) -- nothing to scale")

    rng = random.Random(99)
    samples: List[SizeSample] = []
    for size in sizes:
        # Scale EVERY sequence-like parameter to the same size, not just
        # one. A function comparing two collections (e.g. "is x in
        # list_b") has its real cost living in whichever one is scaled --
        # scaling only the first parameter and leaving a second, equally
        # relevant collection at its edge-case size (often [], i.e. O(1)
        # trivially) silently defeats the whole measurement. Confirmed
        # empirically: an actual O(n) membership-scan -> O(1) set-lookup
        # rewrite measured as ~0.8x ("slower") until this was fixed,
        # because the searched collection was empty in both versions.
        args = [
            generate_sized_value(kinds[p], size, rng) if p in scalable
            else generate_edge_case_values(kinds[p], rng)[0]
            for p in param_order
        ]
        timed_expr = _build_timed_call_expr(func_name, args, repeats)

        orig_result = run_isolated(original, timed_expr, timeout=timeout)
        cand_result = run_isolated(candidate, timed_expr, timeout=timeout)

        if not orig_result.ok or not cand_result.ok:
            return SpeedupResult(False, note=f"execution failed at size {size} -- cannot measure speed of broken code")

        orig_time = (orig_result.return_value[-1] - orig_result.return_value[0]) / repeats
        cand_time = (cand_result.return_value[-1] - cand_result.return_value[0]) / repeats

        samples.append(SizeSample(size=size, original_seconds=orig_time, candidate_seconds=cand_time))

    ratios = [s.original_seconds / max(s.candidate_seconds, 1e-9) for s in samples]
    trend_changed = ratios[-1] > ratios[0] * 2 and ratios[-1] > 1.5

    return SpeedupResult(measured=True, samples=samples, complexity_class_likely_changed=trend_changed)
