"""Tests for each refactor rule module."""
import ast
import pytest
from app.refactor.rules import constant_fold, dead_code, unused_vars, simplify_conditions, loop_opts
from app.refactor.orchestrator import run_all_rules
from app.models import OptimizationLevel


# ── Constant folding ──────────────────────────────────────────────────────

def test_constant_fold_arithmetic():
    tree = ast.parse("x = 2 * 3 + 4")
    new_tree, changes = constant_fold.apply(tree)
    code = ast.unparse(new_tree)
    assert "10" in code
    assert len(changes) > 0


def test_constant_fold_power():
    tree = ast.parse("z = 2 ** 8")
    new_tree, changes = constant_fold.apply(tree)
    assert "256" in ast.unparse(new_tree)


def test_constant_fold_string():
    tree = ast.parse('s = "hello" + " world"')
    new_tree, changes = constant_fold.apply(tree)
    assert "hello world" in ast.unparse(new_tree)


def test_constant_fold_no_divide_by_zero():
    tree = ast.parse("x = 1 / 0")
    new_tree, changes = constant_fold.apply(tree)
    # Should NOT fold — leave original
    assert "1 / 0" in ast.unparse(new_tree)
    assert len(changes) == 0


# ── Dead code ─────────────────────────────────────────────────────────────

def test_dead_code_after_return():
    code = "def f():\n    return 1\n    x = 2\n    y = 3"
    tree = ast.parse(code)
    new_tree, changes = dead_code.apply(tree)
    result = ast.unparse(new_tree)
    assert "x = 2" not in result
    assert len(changes) > 0


def test_dead_code_while_false():
    code = "while False:\n    print('hi')"
    tree = ast.parse(code)
    new_tree, changes = dead_code.apply(tree)
    result = ast.unparse(new_tree)
    assert "while False" not in result
    assert len(changes) > 0


def test_dead_code_if_false():
    code = "if False:\n    x = 1"
    tree = ast.parse(code)
    new_tree, changes = dead_code.apply(tree)
    assert len(changes) > 0


# ── Unused variables ──────────────────────────────────────────────────────

def test_unused_variable_removed():
    code = "def f():\n    x = 5\n    return 42"
    tree = ast.parse(code)
    new_tree, changes = unused_vars.apply(tree)
    result = ast.unparse(new_tree)
    assert "x = 5" not in result
    assert len(changes) > 0


def test_used_variable_kept():
    code = "def f():\n    x = 5\n    return x"
    tree = ast.parse(code)
    new_tree, changes = unused_vars.apply(tree)
    result = ast.unparse(new_tree)
    assert "x = 5" in result
    assert len(changes) == 0


def test_underscore_variable_kept():
    code = "def f():\n    _unused = 5\n    return 42"
    tree = ast.parse(code)
    _, changes = unused_vars.apply(tree)
    assert len(changes) == 0


# ── Condition simplification ──────────────────────────────────────────────

def test_simplify_eq_true_is_advice_when_type_unknown():
    """
    UPDATED: this test previously asserted that `x == True` is rewritten to
    `x`. That rewrite is not safe — `2 == True` is False while `if 2` is
    truthy, so the transform silently changed behaviour for every non-bool
    value. The rule now reports it as advice and leaves the code alone.
    """
    tree = ast.parse("if x == True: pass")
    new_tree, changes = simplify_conditions.apply(tree)
    result = ast.unparse(new_tree)
    assert "== True" in result, "unsafe rewrite must not be applied"
    assert len(changes) > 0, "the issue should still be reported"
    assert all(c.applied is False for c in changes)


def test_simplify_eq_false_is_advice_when_type_unknown():
    """UPDATED for the same reason: `[] == False` is False, but `not []` is True."""
    tree = ast.parse("if x == False: pass")
    new_tree, changes = simplify_conditions.apply(tree)
    result = ast.unparse(new_tree)
    assert "== False" in result
    assert all(c.applied is False for c in changes)


def test_simplify_eq_true_applied_when_provably_bool():
    """When the left side is provably a bool, the rewrite IS safe and is applied."""
    tree = ast.parse("if (a < b) == True: pass")
    new_tree, changes = simplify_conditions.apply(tree)
    result = ast.unparse(new_tree)
    assert "== True" not in result
    assert any(c.applied for c in changes)


def test_redundant_else_after_return():
    code = "def f(x):\n    if x > 0:\n        return 1\n    else:\n        return -1"
    tree = ast.parse(code)
    new_tree, changes = simplify_conditions.apply(tree)
    result = ast.unparse(new_tree)
    assert len(changes) > 0


# ── Loop optimisations ────────────────────────────────────────────────────

def test_list_to_comprehension():
    code = "result = []\nfor x in items:\n    result.append(x * 2)"
    tree = ast.parse(code)
    new_tree, changes = loop_opts.apply(tree)
    result = ast.unparse(new_tree)
    assert "for x in items" in result  # comprehension kept the loop
    assert ".append" not in result
    assert len(changes) > 0


def test_redundant_list_constructor():
    tree = ast.parse("x = list([1, 2, 3])")
    new_tree, changes = loop_opts.apply(tree)
    result = ast.unparse(new_tree)
    assert "list(" not in result
    assert "[1, 2, 3]" in result


# ── Orchestrator ─────────────────────────────────────────────────────────

def test_run_all_rules_medium():
    code = "def f():\n    x = 2 * 3\n    y = 10\n    return 42"
    result = run_all_rules(code, level=OptimizationLevel.MEDIUM)
    assert result.refactored_code is not None
    assert result.unchanged is False
    # constant_fold and unused_var both ran — code is shorter/cleaner
    assert len(result.applied_rules) > 0


def test_run_all_rules_invalid_code():
    result = run_all_rules("def f(\n    pass")
    assert result.unchanged is True
    assert result.refactored_code == "def f(\n    pass"


def test_run_all_rules_no_crash_on_empty():
    result = run_all_rules("")
    assert result.unchanged is True
