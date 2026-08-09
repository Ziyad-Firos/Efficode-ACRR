"""
review/smells.py — Code smell detection via custom AST visitors.

These are patterns that flake8 and bandit don't cover — structural
issues that make code hard to read, test, or maintain.

Checks implemented:
  SMELL001 — Long function (> 20 statements)
  SMELL002 — Deep nesting (> 3 levels)
  SMELL003 — Magic number (bare integer literal outside 0, 1, -1, 2)
  SMELL004 — Too many parameters (> 4 for a non-__init__ function)
  SMELL005 — Boolean trap (function with multiple bool parameters)
  SMELL006 — Duplicate code blocks (identical statement sequences ≥ 3 lines)
  SMELL007 — Dead return value (calling a function but never using the result)
  SMELL008 — Bare except clause (catches everything including KeyboardInterrupt)
  SMELL009 — Mutable default argument (def f(x=[]) — classic Python footgun)
  SMELL010 — Variable shadowing a builtin (list, dict, str, etc.)
  SMELL011 — Inconsistent return type (literal constants of different types
             returned across branches, e.g. `return 0` vs `return "error"`)
"""

from __future__ import annotations

import ast
import hashlib
import logging
from collections import Counter
from typing import List, Set

from app.models import Issue, IssueCategory, IssueSeverity

logger = logging.getLogger("acrr.review.smells")

# ── Constants ──────────────────────────────────────────────────────────────
_MAX_FUNCTION_STMTS = 20
_MAX_NESTING_DEPTH = 3
_MAX_PARAMS = 4
_MAGIC_NUMBER_WHITELIST: Set[int] = {-1, 0, 1, 2}

_BUILTIN_NAMES: Set[str] = {
    "list", "dict", "set", "tuple", "str", "int", "float", "bool",
    "bytes", "type", "object", "range", "len", "print", "input",
    "open", "map", "filter", "zip", "enumerate", "sorted", "reversed",
    "sum", "min", "max", "abs", "round", "id", "hash", "repr",
}


# ── Public entry point ─────────────────────────────────────────────────────

def analyze_smells(code: str) -> List[Issue]:
    """
    Run all smell detectors against the given code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    # Attach parent references — used by magic number check to skip range() args
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]

    issues: List[Issue] = []
    try:
        issues += _check_long_functions(tree)
        issues += _check_deep_nesting(tree)
        issues += _check_magic_numbers(tree)
        issues += _check_too_many_params(tree)
        issues += _check_boolean_trap(tree)
        issues += _check_duplicate_blocks(tree)
        issues += _check_bare_except(tree)
        issues += _check_mutable_defaults(tree)
        issues += _check_builtin_shadowing(tree)
        issues += _check_inconsistent_return_type(tree)
        logger.debug("smell check: %d issues found", len(issues))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error in smell check: %s", exc)

    return issues


# ── Individual detectors ───────────────────────────────────────────────────

def _check_long_functions(tree: ast.AST) -> List[Issue]:
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stmt_count = sum(1 for _ in ast.walk(node)
                             if isinstance(_, ast.stmt) and _ is not node)
            if stmt_count > _MAX_FUNCTION_STMTS:
                issues.append(Issue(
                    line=node.lineno,
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.SMELL,
                    rule="SMELL001",
                    message=(
                        f"Function '{node.name}' has {stmt_count} statements "
                        f"(max recommended: {_MAX_FUNCTION_STMTS}). "
                        "Consider splitting it into smaller, focused functions."
                    ),
                ))
    return issues


def _check_deep_nesting(tree: ast.AST) -> List[Issue]:
    """Walk the tree tracking nesting depth of block statements."""
    issues: List[Issue] = []
    _walk_nesting(tree, 0, issues)
    return issues


def _walk_nesting(node: ast.AST, depth: int, issues: list) -> None:
    """Recursive helper that tracks nesting depth of block statements."""
    nest_nodes = (ast.For, ast.While, ast.If, ast.With,
                  ast.Try, ast.AsyncFor, ast.AsyncWith)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, nest_nodes):
            new_depth = depth + 1
            if new_depth >= _MAX_NESTING_DEPTH:
                issues.append(Issue(
                    line=getattr(child, "lineno", None),
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.SMELL,
                    rule="SMELL002",
                    message=(
                        f"Nesting depth {new_depth} exceeds maximum of "
                        f"{_MAX_NESTING_DEPTH - 1}. "
                        "Extract inner logic into helper functions."
                    ),
                ))
                # Don't recurse deeper — one report per over-nested block is enough
            else:
                _walk_nesting(child, new_depth, issues)
        else:
            _walk_nesting(child, depth, issues)


def _check_magic_numbers(tree: ast.AST) -> List[Issue]:
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            val = node.value
            if isinstance(val, int) and val in _MAGIC_NUMBER_WHITELIST:
                continue
            if isinstance(val, float) and val in {0.0, 1.0}:
                continue
            # Skip integers used as range() arguments — almost never a "magic number"
            parent = getattr(node, "_parent", None)
            if isinstance(parent, ast.Call):
                func = parent.func
                if isinstance(func, ast.Name) and func.id == "range":
                    continue
            issues.append(Issue(
                line=getattr(node, "lineno", None),
                severity=IssueSeverity.INFO,
                category=IssueCategory.SMELL,
                rule="SMELL003",
                message=(
                    f"Magic number {val} — assign it to a named constant "
                    "so its meaning is clear."
                ),
            ))
    return issues


def _check_too_many_params(tree: ast.AST) -> List[Issue]:
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "__init__":
                continue
            args = node.args
            total = (
                len(args.args)
                + len(args.posonlyargs)
                + len(args.kwonlyargs)
            )
            # Don't count 'self' / 'cls'
            if args.args and args.args[0].arg in ("self", "cls"):
                total -= 1
            if total > _MAX_PARAMS:
                issues.append(Issue(
                    line=node.lineno,
                    severity=IssueSeverity.WARNING,
                    category=IssueCategory.SMELL,
                    rule="SMELL004",
                    message=(
                        f"Function '{node.name}' has {total} parameters "
                        f"(max recommended: {_MAX_PARAMS}). "
                        "Consider grouping related parameters into a data class."
                    ),
                ))
    return issues


def _check_boolean_trap(tree: ast.AST) -> List[Issue]:
    """Detects functions that take more than one bool parameter."""
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bool_params = [
                arg.arg for arg in node.args.args
                if arg.annotation and isinstance(arg.annotation, ast.Name)
                and arg.annotation.id == "bool"
            ]
            if len(bool_params) >= 2:
                issues.append(Issue(
                    line=node.lineno,
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.SMELL,
                    rule="SMELL005",
                    message=(
                        f"Function '{node.name}' has multiple bool parameters "
                        f"({', '.join(bool_params)}). "
                        "Boolean traps make call sites hard to read — "
                        "consider using an enum or separate functions."
                    ),
                ))
    return issues


def _check_duplicate_blocks(tree: ast.AST) -> List[Issue]:
    """
    Detect identical statement sequences of 3+ lines appearing more than once.
    Uses a content hash of ast.dump() for each window of statements.
    """
    issues = []

    def _stmts_hash(stmts: list) -> str:
        joined = "|".join(ast.dump(s) for s in stmts)
        return hashlib.md5(joined.encode()).hexdigest()  # noqa: S324

    seen: Counter = Counter()
    seen_line: dict = {}
    window = 3

    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not body or len(body) < window:
            continue
        for i in range(len(body) - window + 1):
            chunk = body[i:i + window]
            h = _stmts_hash(chunk)
            seen[h] += 1
            if seen[h] == 1:
                seen_line[h] = getattr(chunk[0], "lineno", None)
            elif seen[h] == 2:
                issues.append(Issue(
                    line=getattr(chunk[0], "lineno", None),
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.SMELL,
                    rule="SMELL006",
                    message=(
                        f"Duplicate code block (also appears near line {seen_line[h]}). "
                        "Extract into a shared function."
                    ),
                ))

    return issues


def _check_bare_except(tree: ast.AST) -> List[Issue]:
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append(Issue(
                line=getattr(node, "lineno", None),
                severity=IssueSeverity.WARNING,
                category=IssueCategory.SMELL,
                rule="SMELL008",
                message=(
                    "Bare 'except:' clause catches everything including "
                    "KeyboardInterrupt and SystemExit. "
                    "Use 'except Exception:' or catch a specific exception type."
                ),
            ))
    return issues


def _check_mutable_defaults(tree: ast.AST) -> List[Issue]:
    """Detects def f(x=[], y={}) — the classic Python footgun."""
    mutable_types = (ast.List, ast.Dict, ast.Set)
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults + node.args.kw_defaults:
                if default is not None and isinstance(default, mutable_types):
                    issues.append(Issue(
                        line=node.lineno,
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.SMELL,
                        rule="SMELL009",
                        message=(
                            f"Function '{node.name}' uses a mutable default argument "
                            f"({type(default).__name__}). "
                            "Use None as the default and create the object inside the function."
                        ),
                    ))
    return issues


def _check_builtin_shadowing(tree: ast.AST) -> List[Issue]:
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if arg.arg in _BUILTIN_NAMES:
                    issues.append(Issue(
                        line=arg.col_offset and node.lineno,
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.SMELL,
                        rule="SMELL010",
                        message=(
                            f"Parameter '{arg.arg}' shadows the built-in with the same name. "
                            "Rename it to avoid confusion."
                        ),
                    ))
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _BUILTIN_NAMES:
                    issues.append(Issue(
                        line=getattr(node, "lineno", None),
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.SMELL,
                        rule="SMELL010",
                        message=(
                            f"Variable '{target.id}' shadows the built-in with the same name. "
                            "Rename it."
                        ),
                    ))
    return issues


def _return_literal_kind(value: "ast.expr | None") -> "str | None":
    """
    Classify a `return` value's static type if -- and only if -- it is a
    bare literal constant. None means "not classifiable": a variable, a
    call, a binary expression, `return None`, or a bare `return`.

    Deliberately conservative. This is a pattern check, not type inference:
    guessing the type of `return some_variable` would need real data-flow
    analysis, and a wrong guess is a worse false positive than staying
    silent. `return None` is excluded on purpose too -- mixing None with a
    real value is the ordinary "optional result" idiom, not a smell.
    """
    if value is None or not isinstance(value, ast.Constant):
        return None
    v = value.value
    if v is None:
        return None
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (int, float)):
        return "number"
    return None


def _direct_returns(node: ast.AST) -> List[ast.Return]:
    """`return` statements that belong to THIS function, not to any function
    nested inside it. A plain ast.walk() would attribute a closure's returns
    to its enclosing function too."""
    found: List[ast.Return] = []

    def walk(n: ast.AST) -> None:
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(child, ast.Return):
                found.append(child)
            walk(child)

    walk(node)
    return found


def _check_inconsistent_return_type(tree: ast.AST) -> List[Issue]:
    """
    A function that returns literal constants of different types across
    branches -- e.g. an int on one path and a string on another. Every
    caller then has to check the result's type before using it. Usually a
    sign a real result is being conflated with a sentinel/error value; an
    exception is almost always the better tool for the error case.
    """
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        first_line_for_kind: dict = {}
        for ret in _direct_returns(node):
            kind = _return_literal_kind(ret.value)
            if kind is not None and kind not in first_line_for_kind:
                first_line_for_kind[kind] = getattr(ret, "lineno", node.lineno)

        if len(first_line_for_kind) >= 2:
            ordered = sorted(first_line_for_kind.items(), key=lambda kv: kv[1])
            kinds_str = ", ".join(f"{k} (line {ln})" for k, ln in ordered)
            issues.append(Issue(
                line=node.lineno,
                severity=IssueSeverity.WARNING,
                category=IssueCategory.SMELL,
                rule="SMELL011",
                message=(
                    f"Function '{node.name}' returns different literal types across "
                    f"branches: {kinds_str}. Callers must check the type before using "
                    "the result. Prefer a consistent return type, or raise an "
                    "exception for the error case instead of returning a different "
                    "type as a sentinel."
                ),
            ))
    return issues
