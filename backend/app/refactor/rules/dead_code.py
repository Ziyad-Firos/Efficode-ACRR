"""
refactor/rules/dead_code.py — Dead code elimination.

Removes statements that can never be reached at runtime:

  1. Code after return/break/continue/raise inside a block
     def f():
         return 1
         x = 2    ← dead, removed

  2. Trivially-false if branches
     if False:
         do_something()   ← entire block removed

  3. while False loops
     while False:
         ...              ← entire loop removed

  4. Empty try/except blocks that only contain `pass`
     try:
         pass
     except Exception:
         pass             ← removed (no-op)

Safety rules:
  - Never removes code that could have side effects we can't
    statically prove are absent (e.g. function calls in conditions).
  - `if True:` is NOT simplified — the condition could be reassigned
    at a module level or in tests.
  - Only literal `False` and `while False` patterns are removed.
"""

from __future__ import annotations

import ast
import logging
from typing import List

from app.models import AppliedRule

logger = logging.getLogger("acrr.refactor.dead_code")

_TERMINALS = (ast.Return, ast.Break, ast.Continue, ast.Raise)


class _DeadCodeRemover(ast.NodeTransformer):

    def __init__(self) -> None:
        self.changes: List[AppliedRule] = []

    # ── 1. Statements after a terminal ──────────────────────────────────
    def _trim_after_terminal(self, stmts: list) -> list:
        """Remove statements following a terminal in a flat statement list."""
        result = []
        for stmt in stmts:
            result.append(stmt)
            if isinstance(stmt, _TERMINALS):
                # Check if there are more statements after this
                remaining = stmts[len(result):]
                if remaining:
                    self.changes.append(AppliedRule(
                        rule="dead_code_after_terminal",
                        line=getattr(remaining[0], "lineno", None),
                        description=(
                            f"Removed {len(remaining)} unreachable statement(s) "
                            f"after '{type(stmt).__name__.lower()}' on line {getattr(stmt, 'lineno', '?')}."
                        ),
                    ))
                break
        return result

    def _visit_body(self, stmts: list) -> list:
        visited = [self.visit(s) for s in stmts]
        # Filter out None (nodes the transformer removed)
        visited = [s for s in visited if s is not None]
        return self._trim_after_terminal(visited)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = self._visit_body(node.body)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_For(self, node: ast.For) -> ast.AST:
        node.body = self._visit_body(node.body)
        self.generic_visit(node)
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        # ── 3. while False ──────────────────────────────────────────────
        if isinstance(node.test, ast.Constant) and node.test.value is False:
            self.changes.append(AppliedRule(
                rule="dead_code_while_false",
                line=node.lineno,
                description="Removed unreachable 'while False:' loop.",
            ))
            return None  # type: ignore[return-value]  # NodeTransformer allows None to delete

        node.body = self._visit_body(node.body)
        self.generic_visit(node)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        # ── 2. if False ─────────────────────────────────────────────────
        if isinstance(node.test, ast.Constant) and node.test.value is False:
            if node.orelse:
                self.changes.append(AppliedRule(
                    rule="dead_code_if_false",
                    line=node.lineno,
                    description="Removed 'if False:' branch; kept else branch.",
                ))
                if len(node.orelse) == 1:
                    return self.visit(node.orelse[0])
                node.body = []
                node.test = ast.Constant(value=True)
                return node
            else:
                self.changes.append(AppliedRule(
                    rule="dead_code_if_false",
                    line=node.lineno,
                    description="Removed entire 'if False:' block.",
                ))
                # Use Pass instead of None to avoid empty parent bodies
                new = ast.Pass()
                ast.copy_location(new, node)
                return new

        node.body = self._visit_body(node.body)
        if node.orelse:
            node.orelse = self._visit_body(node.orelse)
        return node

    # ── 4. Empty try/except ─────────────────────────────────────────────
    def visit_Try(self, node: ast.Try) -> ast.AST:
        self.generic_visit(node)

        def _is_only_pass(stmts) -> bool:
            return all(isinstance(s, ast.Pass) for s in stmts) if stmts else True

        finally_body = getattr(node, "finalbody", []) or []

        if (
            _is_only_pass(node.body)
            and all(_is_only_pass(h.body) for h in node.handlers)
            and _is_only_pass(finally_body)
        ):
            self.changes.append(AppliedRule(
                rule="dead_code_empty_try",
                line=node.lineno,
                description="Removed empty try/except block containing only 'pass'.",
            ))
            # Return a Pass node instead of None — deleting the node entirely
            # can leave a parent body empty (invalid Python: empty function body).
            new = ast.Pass()
            ast.copy_location(new, node)
            return new

        return node


# ── Public API ─────────────────────────────────────────────────────────────

def apply(tree: ast.AST) -> tuple[ast.AST, List[AppliedRule]]:
    """
    Apply dead code elimination to the given AST.

    Returns:
        (transformed_tree, list_of_applied_rules)
    """
    remover = _DeadCodeRemover()
    new_tree = remover.visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree, remover.changes
