"""
refactor/rules.py — Rule-based refactoring orchestrator.

This is the single public entry point for the rule-based refactor engine.
It calls each sub-rule module in a defined order, collects all applied
changes, and returns the final refactored code.

Pipeline order (matters — each pass feeds into the next):
  1. constant_fold      — fold numeric/string literals first
  2. dead_code          — remove unreachable blocks
  3. unused_vars        — remove assignments never read
  4. simplify_conditions — clean up boolean comparisons
  5. loop_opts          — comprehensions, enumerate, redundant constructors

Each module exposes a single apply(tree) → (tree, changes) signature.
The orchestrator threads the tree through each module sequentially.

If a module raises for any reason, the error is logged and that module's
pass is skipped — the tree from the previous pass is used instead.
This means a bug in one rule NEVER breaks the whole refactor pipeline.
"""

from __future__ import annotations

import ast
import copy
import logging
from dataclasses import dataclass, field
from typing import List

from app.models import AppliedRule, OptimizationLevel
from app.parser import safe_unparse

# Import each rule module
from app.refactor.rules import constant_fold  # type: ignore[attr-defined]
from app.refactor.rules import dead_code  # type: ignore[attr-defined]
from app.refactor.rules import unused_vars  # type: ignore[attr-defined]
from app.refactor.rules import simplify_conditions  # type: ignore[attr-defined]
from app.refactor.rules import loop_opts  # type: ignore[attr-defined]

logger = logging.getLogger("acrr.refactor.rules")


@dataclass
class RuleResult:
    """Result returned by run_all_rules()."""
    refactored_code: str
    applied_rules: List[AppliedRule] = field(default_factory=list)
    unchanged: bool = False  # True when no rules produced any changes


# Which rules run at each optimization level
_RULES_BY_LEVEL: dict[str, list] = {
    OptimizationLevel.LOW: [
        constant_fold,
        dead_code,
    ],
    # MEDIUM is the default level the frontend sends, so anything omitted
    # here is effectively dead in normal use. loop_opts used to be HIGH-only,
    # which meant the loop optimisation feature never ran for any default
    # request — and so was never exercised or tested.
    OptimizationLevel.MEDIUM: [
        constant_fold,
        dead_code,
        unused_vars,
        simplify_conditions,
        loop_opts,
    ],
    OptimizationLevel.HIGH: [
        constant_fold,
        dead_code,
        unused_vars,
        simplify_conditions,
        loop_opts,
    ],
}


def run_all_rules(code: str, level: OptimizationLevel = OptimizationLevel.MEDIUM) -> RuleResult:
    """
    Run all rule-based refactoring passes on the given code.

    Args:
        code:  Python source string (already syntax-validated by main.py).
        level: Optimization aggressiveness (low/medium/high).

    Returns:
        RuleResult with the final code and list of applied changes.
        If something goes wrong, returns the original code unchanged.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        logger.error("Could not parse code in rule engine: %s", exc)
        return RuleResult(refactored_code=code, unchanged=True)

    all_changes: List[AppliedRule] = []
    rules = _RULES_BY_LEVEL.get(level, _RULES_BY_LEVEL[OptimizationLevel.MEDIUM])

    for rule_module in rules:
        module_name = rule_module.__name__.rsplit(".", 1)[-1]
        try:
            tree_before = copy.deepcopy(tree)  # rollback point
            tree, changes = rule_module.apply(tree)
            all_changes.extend(changes)
            if changes:
                logger.debug("%s: %d change(s) applied", module_name, len(changes))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Rule module '%s' raised an exception — skipping: %s",
                module_name, exc,
            )
            tree = tree_before  # roll back to the state before this module

    # Convert back to source
    refactored = safe_unparse(tree)
    if refactored is None:
        logger.error("ast.unparse failed — returning original code")
        return RuleResult(refactored_code=code, unchanged=True)

    return RuleResult(
        refactored_code=refactored,
        applied_rules=all_changes,
        unchanged=refactored == code,
    )
