"""
refactor/rules/loop_opts.py — Safe loop optimizations.

Patterns handled
----------------

  1. range(len(x)) enumeration                         [SUGGESTION ONLY]
     for i in range(len(items)):
         print(items[i])
     Reported as advice. Not auto-rewritten, because rewriting requires
     tracking every use of items[i] across the body.

  2. Accumulator loop → list comprehension             [REWRITE]
     result = []
     for x in items:
         result.append(x * 2)
     → result = [x * 2 for x in items]

  3. Redundant list() around a list literal            [REWRITE]
     x = list([1, 2, 3])  →  x = [1, 2, 3]

  4. Redundant dict() around a dict literal            [REWRITE]
     x = dict({"a": 1})   →  x = {"a": 1}


Safety rules for pattern 2
--------------------------
A for-loop and a list comprehension are NOT interchangeable. Three
differences will silently produce broken or wrong code. Each is guarded
below, and each guard corresponds to a case verified to break at runtime:

  a) The accumulator is invisible inside the comprehension.
         result = []
         for x in items:
             result.append(len(result))     # counts what is there so far
     Rewriting gives `[len(result) for x in items]`, which raises
     NameError — a comprehension has its own scope and `result` is not
     bound inside it. Guard: refuse if the accumulator name appears
     anywhere in the appended expression or the iterable.

  b) The loop variable escapes a for-loop, but not a comprehension.
         for x in items:
             result.append(x * 2)
         print(x)                            # legal after a for-loop
     After rewriting, `x` is undefined and `print(x)` raises NameError.
     Guard: refuse if any loop target is referenced later in the block.

  c) The loop target may shadow the accumulator.
         result = []
         for result in items:
             result.append(1)
     Here `result` is rebound each iteration, so the appends land on the
     items rather than the accumulator. Guard: refuse if the target
     shadows the accumulator name.

Plus the structural guards: exactly one body statement, that statement is
`<accumulator>.append(<single expr>)`, and no else clause.

Known limitation: liveness for guard (b) is checked against the rest of the
enclosing block. Since _BlockOptimizer only rewrites Module and FunctionDef
bodies, that block is the whole function — but a loop nested inside an `if`
is not rewritten at all, so the conservative direction is preserved.
"""

from __future__ import annotations

import ast
import logging
from typing import List, Optional, Set

from app.models import AppliedRule

logger = logging.getLogger("acrr.refactor.loop_opts")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _names_referenced(node: ast.AST) -> Set[str]:
    """Every identifier mentioned inside node, read or written."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _target_names(target: ast.expr) -> Optional[Set[str]]:
    """
    Names bound by a for-loop target.

    Returns None for targets we do not fully understand (attribute or
    subscript targets such as `for obj.attr in ...`), which is a signal to
    decline the rewrite rather than guess at it.
    """
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: Set[str] = set()
        for element in target.elts:
            sub = _target_names(element)
            if sub is None:
                return None
            names |= sub
        return names
    return None


def _is_range_len_loop(node: ast.For) -> bool:
    """True if the loop iterable is exactly range(len(something))."""
    iterable = node.iter
    if not (isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name)):
        return False
    if iterable.func.id != "range" or len(iterable.args) != 1:
        return False
    inner = iterable.args[0]
    return (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "len"
    )


# ---------------------------------------------------------------------------
# Patterns 1, 3, 4 — single-node transforms
# ---------------------------------------------------------------------------

class _LoopOptimizer(ast.NodeTransformer):

    def __init__(self) -> None:
        self.changes: List[AppliedRule] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        if not isinstance(node.func, ast.Name):
            return node

        # list([...]) → [...]
        if (
            node.func.id == "list"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.List)
        ):
            self.changes.append(AppliedRule(
                rule="redundant_list_constructor",
                line=node.lineno,
                description="Removed a redundant list([...]) wrapper — the literal is already a list.",
            ))
            return node.args[0]

        # dict({...}) → {...}
        if (
            node.func.id == "dict"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Dict)
        ):
            self.changes.append(AppliedRule(
                rule="redundant_dict_constructor",
                line=node.lineno,
                description="Removed a redundant dict({...}) wrapper — the literal is already a dict.",
            ))
            return node.args[0]

        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)

        if _is_range_len_loop(node):
            self.changes.append(AppliedRule(
                rule="range_len_to_enumerate",
                line=node.lineno,
                description=(
                    "Consider 'for i, item in enumerate(x):' instead of "
                    "'for i in range(len(x)):' — it avoids repeated index lookups."
                ),
                applied=False,   # advice only; the code was not modified
            ))

        return node


# ---------------------------------------------------------------------------
# Pattern 2 — accumulator loop → list comprehension
# ---------------------------------------------------------------------------

def _match_append_loop(
    list_name: str,
    node: ast.AST,
    later_statements: List[ast.stmt],
) -> Optional[ast.ListComp]:
    """
    Return an equivalent ListComp for `for t in it: list_name.append(expr)`,
    or None if any safety guard fails.

    `later_statements` is the rest of the enclosing block, needed to decide
    whether the loop variable is still live after the loop.
    """
    if not isinstance(node, ast.For):
        return None
    if node.orelse or len(node.body) != 1:
        return None

    body_stmt = node.body[0]
    if not isinstance(body_stmt, ast.Expr):
        return None

    call = body_stmt.value
    if not isinstance(call, ast.Call):
        return None

    # must be exactly `list_name.append(<one positional arg>)`
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "append":
        return None
    if not isinstance(call.func.value, ast.Name) or call.func.value.id != list_name:
        return None
    if len(call.args) != 1 or call.keywords:
        return None

    elt_expr = call.args[0]
    iter_expr = node.iter

    # Guard (a): the accumulator is not visible inside a comprehension.
    if list_name in _names_referenced(elt_expr):
        logger.debug(
            "loop_to_comprehension declined: '%s' is read in the appended expression, "
            "which a comprehension cannot see", list_name,
        )
        return None
    if list_name in _names_referenced(iter_expr):
        logger.debug("loop_to_comprehension declined: '%s' appears in the iterable", list_name)
        return None

    # Target must be a plain name or a tuple/list of names.
    targets = _target_names(node.target)
    if targets is None:
        logger.debug("loop_to_comprehension declined: unsupported loop target")
        return None

    # Guard (c): the target must not shadow the accumulator.
    if list_name in targets:
        logger.debug("loop_to_comprehension declined: loop target shadows '%s'", list_name)
        return None

    # Guard (b): the loop variable must be dead after the loop.
    for stmt in later_statements:
        if targets & _names_referenced(stmt):
            logger.debug(
                "loop_to_comprehension declined: loop variable %s is used after the "
                "loop; a comprehension would not bind it", targets,
            )
            return None

    comp = ast.ListComp(
        elt=elt_expr,
        generators=[ast.comprehension(
            target=node.target, iter=iter_expr, ifs=[], is_async=0,
        )],
    )
    ast.copy_location(comp, node)
    ast.fix_missing_locations(comp)
    return comp


class _BlockOptimizer(ast.NodeTransformer):
    """
    Handles patterns spanning consecutive statements — `result = []` followed
    by a loop that appends to it.
    """

    def __init__(self, changes: List[AppliedRule]) -> None:
        self.changes = changes

    def _optimize_body(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            if (
                i + 1 < len(stmts)
                and isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.List)
                and len(stmt.value.elts) == 0
            ):
                list_name = stmt.targets[0].id
                comp = _match_append_loop(list_name, stmts[i + 1], stmts[i + 2:])
                if comp is not None:
                    new_assign = ast.Assign(
                        targets=[ast.Name(id=list_name, ctx=ast.Store())],
                        value=comp,
                        lineno=stmt.lineno,
                        col_offset=stmt.col_offset,
                    )
                    ast.fix_missing_locations(new_assign)
                    self.changes.append(AppliedRule(
                        rule="loop_to_comprehension",
                        line=stmt.lineno,
                        description=(
                            f"Replaced the '{list_name} = []' + append loop with a list "
                            f"comprehension — same result, less bookkeeping."
                        ),
                    ))
                    out.append(new_assign)
                    i += 2
                    continue
            out.append(stmt)
            i += 1
        return out

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = self._optimize_body(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._optimize_body(node.body)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply(tree: ast.AST) -> tuple[ast.AST, List[AppliedRule]]:
    """
    Apply loop optimizations.

    Returns (transformed_tree, applied_rules). Entries with applied=False are
    advice that did not modify the code.
    """
    changes: List[AppliedRule] = []

    opt = _LoopOptimizer()
    tree = opt.visit(tree)
    changes.extend(opt.changes)

    block_opt = _BlockOptimizer(changes)
    tree = block_opt.visit(tree)

    ast.fix_missing_locations(tree)
    return tree, changes
