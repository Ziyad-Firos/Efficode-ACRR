"""
refactor/rules/constant_fold.py — Constant folding.

Evaluates binary expressions whose operands are both compile-time
constants and replaces them with the computed result.

Examples:
  x = 2 * 3 + 4        →  x = 10
  y = 10 / 2           →  y = 5.0
  z = 2 ** 8           →  z = 256
  s = "hello" + " world" → s = "hello world"
  b = True and False   →  b = False

Safety rule: only numeric and string constants are folded.
Expressions that could raise (e.g. division by zero) are caught
and left unchanged.
"""

from __future__ import annotations

import ast
import logging
from typing import List

from app.models import AppliedRule

logger = logging.getLogger("acrr.refactor.constant_fold")


class _ConstantFolder(ast.NodeTransformer):
    """AST transformer that folds constant binary/unary expressions."""

    def __init__(self) -> None:
        self.changes: List[AppliedRule] = []

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        # Recurse into children first (bottom-up)
        self.generic_visit(node)

        left = node.left
        right = node.right

        if not (isinstance(left, ast.Constant) and isinstance(right, ast.Constant)):
            return node

        lv, rv = left.value, right.value

        # Only fold numeric and string constants
        if not isinstance(lv, (int, float, complex, str)):
            return node
        if not isinstance(rv, (int, float, complex, str)):
            return node

        try:
            result = _eval_binop(node.op, lv, rv)
        except (ZeroDivisionError, TypeError, ValueError):
            return node  # leave unchanged — don't produce broken code

        new_node = ast.Constant(value=result)
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)

        self.changes.append(AppliedRule(
            rule="constant_fold",
            line=node.lineno,
            description=(
                f"Folded constant expression "
                f"'{ast.unparse(left)} {_op_symbol(node.op)} {ast.unparse(right)}' "
                f"→ {result!r}"
            ),
        ))
        return new_node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)

        if not isinstance(node.operand, ast.Constant):
            return node

        val = node.operand.value
        if not isinstance(val, (int, float, complex)):
            return node

        try:
            result = _eval_unaryop(node.op, val)
        except (TypeError, ValueError):
            return node

        new_node = ast.Constant(value=result)
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)

        self.changes.append(AppliedRule(
            rule="constant_fold",
            line=node.lineno,
            description=f"Folded unary expression → {result!r}",
        ))
        return new_node


def _eval_binop(op: ast.operator, lv, rv):
    """Evaluate a binary operation given Python values. Raises on error."""
    ops = {
        ast.Add:      lambda a, b: a + b,
        ast.Sub:      lambda a, b: a - b,
        ast.Mult:     lambda a, b: a * b,
        ast.Div:      lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod:      lambda a, b: a % b,
        ast.Pow:      lambda a, b: a ** b,
        ast.BitAnd:   lambda a, b: a & b,
        ast.BitOr:    lambda a, b: a | b,
        ast.BitXor:   lambda a, b: a ^ b,
        ast.LShift:   lambda a, b: a << b,
        ast.RShift:   lambda a, b: a >> b,
    }
    fn = ops.get(type(op))
    if fn is None:
        raise TypeError(f"Unsupported op: {type(op)}")
    return fn(lv, rv)


def _eval_unaryop(op: ast.unaryop, val):
    ops = {
        ast.UAdd:  lambda a: +a,
        ast.USub:  lambda a: -a,
        ast.Invert: lambda a: ~a,
        ast.Not:   lambda a: not a,
    }
    fn = ops.get(type(op))
    if fn is None:
        raise TypeError(f"Unsupported unary op: {type(op)}")
    return fn(val)


def _op_symbol(op: ast.operator) -> str:
    symbols = {
        ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
        ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
        ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    }
    return symbols.get(type(op), "?")


# ── Public API ─────────────────────────────────────────────────────────────

def apply(tree: ast.AST) -> tuple[ast.AST, List[AppliedRule]]:
    """
    Apply constant folding to the given AST.

    Returns:
        (transformed_tree, list_of_applied_rules)
    """
    folder = _ConstantFolder()
    new_tree = folder.visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree, folder.changes
