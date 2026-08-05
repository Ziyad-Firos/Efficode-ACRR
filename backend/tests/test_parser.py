"""Tests for the safe parser wrapper."""
import pytest
from app.parser import parse_code, is_valid_python, safe_unparse
import ast


def test_valid_code():
    r = parse_code("x = 1 + 2")
    assert r.valid is True
    assert r.tree is not None
    assert r.error is None


def test_syntax_error():
    r = parse_code("def f(\n    pass")
    assert r.valid is False
    assert r.error is not None
    assert "line" in r.error.lower() or r.error_line is not None


def test_empty_code():
    r = parse_code("")
    assert r.valid is False
    assert "empty" in r.error.lower()


def test_whitespace_only():
    r = parse_code("   \n\t  ")
    assert r.valid is False


def test_is_valid_python():
    assert is_valid_python("x = 1") is True
    assert is_valid_python("def f:") is False


def test_safe_unparse():
    tree = ast.parse("x = 1 + 2")
    result = safe_unparse(tree)
    assert result is not None
    assert "x" in result
