"""
refactor/rules/simplify_conditions.py — Condition simplifications.

Patterns handled
----------------
  1. x == True   →  x          [only when x is provably a bool — see below]
  2. x == False  →  not x      [same]
  3. x != True   →  not x      [same]
  4. x != False  →  x          [same]
  5. redundant else after return
         if cond:
             return a
         else:
             return b
     →   if cond:
             return a
         return b


Why patterns 1–4 are guarded
----------------------------
`if x == True:` and `if x:` are NOT the same thing, and rewriting one into
the other silently changes behaviour. Verified divergences:

    x = 2      `2 == True`  is False  but  `if 2`      is truthy
    x = [1]    `[1] == True` is False but  `if [1]`    is truthy
    x = "a"    `"a" == True` is False but  `if "a"`    is truthy
    x = []     `[] == False` is False but  `if not []` is True
    x = None   `None == False` is False but `if not None` is True

`is True` has exactly the same problem — `2 is True` is False while `if 2`
is truthy — so identity comparisons are guarded too.

The comparison is therefore rewritten only when the left operand is
*provably* a bool: another comparison, a `not` expression, a boolean
literal, or a call to a builtin with a bool return type. In every other
case the issue is still reported, but as advice (`applied=False`) rather
than an automatic rewrite — the user may know `x` is a bool, but this
module cannot, and guessing wrong corrupts their code.

This mirrors flake8's E712, which flags the comparison but does not fix it.
"""

from __future__ import annotations

import ast
import logging
from typing import List

from app.models import AppliedRule

logger = logging.getLogger("acrr.refactor.simplify_conditions")

# Builtins that are documented to return a bool.
_BOOL_RETURNING_BUILTINS = {
    "isinstance", "issubclass", "bool", "callable", "hasattr",
    "any", "all",
}


def _is_provably_bool(node: ast.expr) -> bool:
    """
    True when the expression is guaranteed to evaluate to a real bool.

    Deliberately conservative. `a and b` is NOT included: `and`/`or` return
    one of their operands, not a bool, so `(a and b) == True` is not
    equivalent to `a and b`.
    """
    if isinstance(node, ast.Compare):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in _BOOL_RETURNING_BUILTINS
    return False


class _ConditionSimplifier(ast.NodeTransformer):

    def __init__(self) -> None:
        self.changes: List[AppliedRule] = []

    # ── Patterns 1–4 ────────────────────────────────────────────────────────
    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)

        if len(node.ops) != 1 or len(node.comparators) != 1:
            return node

        op = node.ops[0]
        right = node.comparators[0]

        # right side must be the literal True or False (not 1/0 — `is` on
        # ints is a different question, and `1 == True` is True)
        if not (isinstance(right, ast.Constant) and isinstance(right.value, bool)):
            return node

        compares_true = right.value is True
        is_equality = isinstance(op, (ast.Eq, ast.Is))
        is_inequality = isinstance(op, (ast.NotEq, ast.IsNot))
        if not (is_equality or is_inequality):
            return node

        literal = "True" if compares_true else "False"
        operator = {ast.Eq: "==", ast.Is: "is",
                    ast.NotEq: "!=", ast.IsNot: "is not"}[type(op)]
        original = f"x {operator} {literal}"

        # `== True` and `!= False` keep the value; `== False` and `!= True` negate it
        keeps_value = (is_equality and compares_true) or (is_inequality and not compares_true)
        replacement = "x" if keeps_value else "not x"

        if not _is_provably_bool(node.left):
            # Report, but do not rewrite — see the module docstring.
            self.changes.append(AppliedRule(
                rule="simplify_conditions",
                line=getattr(node, "lineno", None),
                description=(
                    f"'{original}' can usually be written as '{replacement}'. "
                    f"Not applied automatically: if the left side is not a bool "
                    f"this would change behaviour (2 == True is False, but 'if 2' "
                    f"is true). Apply it yourself if you know the value is a bool."
                ),
                applied=False,
            ))
            return node

        self.changes.append(AppliedRule(
            rule="simplify_conditions",
            line=getattr(node, "lineno", None),
            description=(
                f"Simplified '{original}' to '{replacement}' — the left side is "
                f"already a boolean expression."
            ),
        ))

        if keeps_value:
            return node.left

        new = ast.UnaryOp(op=ast.Not(), operand=node.left)
        ast.copy_location(new, node)
        ast.fix_missing_locations(new)
        return new

    # ── Pattern 5: redundant else after return ─────────────────────────────
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body = self._remove_redundant_else(node.body)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def _remove_redundant_else(self, stmts: list) -> list:
        """
        Hoist the else body out when the if body always returns.

        Safe because a body whose last statement is a `return` cannot fall
        through to the code after the if. Only applied when the if body is
        terminal — a body ending in `break`/`continue` is left alone, since
        it terminates the loop iteration rather than the function.
        """
        result = []
        for stmt in stmts:
            if (
                isinstance(stmt, ast.If)
                and stmt.orelse
                and stmt.body
                and isinstance(stmt.body[-1], ast.Return)
            ):
                hoisted = stmt.orelse
                stmt.orelse = []
                self.changes.append(AppliedRule(
                    rule="simplify_conditions",
                    line=getattr(stmt, "lineno", None),
                    description=(
                        "Removed a redundant 'else' after a 'return' — the if branch "
                        "already exits, so the else body can be un-indented."
                    ),
                ))
                result.append(stmt)
                result.extend(hoisted)
            else:
                result.append(stmt)
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply(tree: ast.AST) -> tuple[ast.AST, List[AppliedRule]]:
    """
    Apply condition simplifications.

    Returns (transformed_tree, applied_rules). Entries with applied=False are
    advice that did not modify the code.
    """
    simplifier = _ConditionSimplifier()
    new_tree = simplifier.visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree, simplifier.changes
