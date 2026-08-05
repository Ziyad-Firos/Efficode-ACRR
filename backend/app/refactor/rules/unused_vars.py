"""
refactor/rules/unused_vars.py — Unused variable removal.

Finds local variables that are assigned but never read, then
removes the assignment statement.

Rules:
  - Only removes *local* assignments (inside a function body).
    Module-level assignments are skipped — they may be public API.
  - Variables starting with '_' are intentionally ignored
    (Python convention: _ means "I know this is unused").
  - Loop variables are skipped (for x in ... — x is considered used
    by the loop machinery even if not referenced in the body).
  - Variables used in closures (nested functions) are preserved.
  - Augmented assignments (x += 1) are NOT removed even if x is
    never read afterwards — too risky without full dataflow analysis.

Example:
    def f():
        x = 5        ← assigned but never read → removed
        y = x + 1    ← y never read → removed
        return 42

    becomes:
    def f():
        return 42
"""

from __future__ import annotations

import ast
import logging
from typing import List, Set

from app.models import AppliedRule

logger = logging.getLogger("acrr.refactor.unused_vars")


class _UsageCollector(ast.NodeVisitor):
    """Collect all Name nodes that are *read* (Load context) inside a scope."""

    def __init__(self) -> None:
        self.used: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        self.generic_visit(node)


class _FunctionCleaner(ast.NodeTransformer):
    """
    For each FunctionDef, collect used names, then remove assignments
    to names that are never read.
    """

    def __init__(self) -> None:
        self.changes: List[AppliedRule] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._clean_function(node)
        self.generic_visit(node)   # recurse into nested functions
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def _clean_function(self, func_node) -> None:
        # 1. Collect all names that are READ anywhere in this function
        collector = _UsageCollector()
        collector.visit(func_node)
        used = collector.used

        # 2. Collect loop target variables — never remove these
        loop_vars: Set[str] = set()
        for node in ast.walk(func_node):
            if isinstance(node, (ast.For, ast.AsyncFor)):
                for n in ast.walk(node.target):
                    if isinstance(n, ast.Name):
                        loop_vars.add(n.id)

        # 3. Collect function parameters — never remove these
        params: Set[str] = set()
        for arg in (func_node.args.args
                    + func_node.args.posonlyargs
                    + func_node.args.kwonlyargs):
            params.add(arg.arg)
        if func_node.args.vararg:
            params.add(func_node.args.vararg.arg)
        if func_node.args.kwarg:
            params.add(func_node.args.kwarg.arg)

        # 4. Walk the body and remove assignments to unused names
        func_node.body = self._remove_from_stmts(
            func_node.body, used, loop_vars, params
        )

    def _remove_from_stmts(self, stmts: list, used, loop_vars, params) -> list:
        new_stmts = []
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                removed = self._try_remove_assign(stmt, used, loop_vars, params)
                if removed:
                    continue   # drop this statement

            # Recurse into nested blocks
            if hasattr(stmt, "body"):
                stmt.body = self._remove_from_stmts(stmt.body, used, loop_vars, params)
            if hasattr(stmt, "orelse") and stmt.orelse:
                stmt.orelse = self._remove_from_stmts(stmt.orelse, used, loop_vars, params)
            if hasattr(stmt, "handlers"):
                for handler in stmt.handlers:
                    handler.body = self._remove_from_stmts(handler.body, used, loop_vars, params)

            new_stmts.append(stmt)
        return new_stmts

    def _try_remove_assign(
        self, node: ast.Assign, used, loop_vars, params
    ) -> bool:
        """
        Return True if this assignment should be removed.
        Only removes simple `name = expr` where the RHS has no side effects.
        """
        if len(node.targets) != 1:
            return False

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return False

        name = target.id

        # Preserve underscore-prefixed names (intentionally unused convention)
        if name.startswith("_"):
            return False

        # Preserve loop vars and params
        if name in loop_vars or name in params:
            return False

        # Only remove if never read
        if name in used:
            return False

        # Don't remove if RHS has side effects (function calls, attribute access, etc.)
        if _has_side_effects(node.value):
            return False

        self.changes.append(AppliedRule(
            rule="unused_variable",
            line=node.lineno,
            description=f"Removed unused variable '{name}' (assigned but never read).",
        ))
        return True


def _has_side_effects(node: ast.AST) -> bool:
    """
    Conservative check: return True if the expression might have side effects.
    We only allow removal if the RHS is a simple constant or name lookup.
    """
    for n in ast.walk(node):
        if isinstance(n, (ast.Call, ast.Await, ast.YieldFrom, ast.Yield)):
            return True
        if isinstance(n, (ast.Attribute, ast.Subscript)):
            return True   # getattr/getitem can have __get__ side effects
    return False


# ── Public API ─────────────────────────────────────────────────────────────

def apply(tree: ast.AST) -> tuple[ast.AST, List[AppliedRule]]:
    """
    Remove unused local variable assignments from the AST.

    Returns:
        (transformed_tree, list_of_applied_rules)
    """
    cleaner = _FunctionCleaner()
    new_tree = cleaner.visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree, cleaner.changes
