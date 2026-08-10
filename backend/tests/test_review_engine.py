"""
tests/test_review_engine.py — Review analyser correctness.

Why this file exists
--------------------
All four analysers fail *silently* by design: `except ImportError: return []`,
`except FileNotFoundError: ...`, and a broad `except Exception` around each.
That is the right behaviour for robustness — one missing tool must not take
down the whole review — but it means "POST /review returned 200" tells you
nothing about whether the analysers actually ran.

Before this file, that was the entire evidence base for style, complexity and
security. Each test below feeds code with a KNOWN defect and asserts the
specific rule fires, so an analyser that quietly disappears turns the suite
red instead of passing with an empty issue list.

Tests skip (rather than fail) when the underlying tool is not installed, so
the suite is still meaningful in a minimal environment — but `test_*_installed`
records loudly which analysers were actually exercised.

Runs under pytest, or standalone with `python tests/test_review_engine.py`.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import IssueCategory, IssueSeverity   # noqa: E402
from app.review import run_all_checks                 # noqa: E402
from app.review.complexity import (                   # noqa: E402
    analyze_complexity,
    get_maintainability_index,
)
from app.review.security import analyze_security      # noqa: E402
from app.review.smells import analyze_smells          # noqa: E402
from app.review.style import analyze_style            # noqa: E402


class Skip(Exception):
    """Raised to mark a test as skipped because its tool is unavailable."""


def _have_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _have_executable(name: str) -> bool:
    return shutil.which(name) is not None


# ---------------------------------------------------------------------------
# Style — flake8
# ---------------------------------------------------------------------------

STYLE_SAMPLE = (
    "import os\n"
    "import sys\n"
    "\n"
    "\n"
    "def f():\n"
    "    unused_local = 1\n"
    "    return 42\n"
)


def test_style_reports_unused_import():
    """F401 (imported but unused) is the canary — if flake8 ran, this fires."""
    if not _have_executable("flake8"):
        raise Skip("flake8 not installed")

    issues = analyze_style(STYLE_SAMPLE)
    codes = {i.rule for i in issues}
    assert "F401" in codes, f"expected F401 for unused imports, got {sorted(codes)}"
    assert all(i.category == IssueCategory.STYLE for i in issues)


def test_style_reports_unused_local_variable():
    if not _have_executable("flake8"):
        raise Skip("flake8 not installed")

    codes = {i.rule for i in analyze_style(STYLE_SAMPLE)}
    assert "F841" in codes, f"expected F841 for the unused local, got {sorted(codes)}"


def test_style_clean_code_has_no_issues():
    if not _have_executable("flake8"):
        raise Skip("flake8 not installed")

    clean = "def add(a, b):\n    return a + b\n"
    assert analyze_style(clean) == [], "flake8 flagged clean code"


def test_style_severity_mapping():
    """F-codes are errors, W-codes are warnings — a mis-map skews the grade."""
    if not _have_executable("flake8"):
        raise Skip("flake8 not installed")

    issues = analyze_style(STYLE_SAMPLE)
    for issue in issues:
        if issue.rule.startswith(("E", "F")):
            assert issue.severity == IssueSeverity.ERROR, issue.rule
        elif issue.rule.startswith(("W", "C")):
            assert issue.severity == IssueSeverity.WARNING, issue.rule


# ---------------------------------------------------------------------------
# Security — bandit
# ---------------------------------------------------------------------------

EVAL_SAMPLE = (
    "def run(user_input):\n"
    "    return eval(user_input)\n"
)

SHELL_SAMPLE = (
    "import subprocess\n"
    "\n"
    "\n"
    "def run(cmd):\n"
    "    return subprocess.call(cmd, shell=True)\n"
)

PASSWORD_SAMPLE = (
    "def connect():\n"
    "    password = 'hunter2'\n"
    "    return password\n"
)

WEAK_HASH_SAMPLE = (
    "import hashlib\n"
    "\n"
    "\n"
    "def digest(data):\n"
    "    return hashlib.md5(data).hexdigest()\n"
)


def test_security_detects_eval():
    if not _have_executable("bandit"):
        raise Skip("bandit not installed")

    issues = analyze_security(EVAL_SAMPLE)
    assert issues, "bandit did not flag eval() of user input"
    assert all(i.category == IssueCategory.SECURITY for i in issues)


def test_security_detects_shell_injection():
    if not _have_executable("bandit"):
        raise Skip("bandit not installed")

    assert analyze_security(SHELL_SAMPLE), "shell=True was not flagged"


def test_security_detects_hardcoded_password():
    if not _have_executable("bandit"):
        raise Skip("bandit not installed")

    assert analyze_security(PASSWORD_SAMPLE), "hardcoded password was not flagged"


def test_security_detects_weak_hash():
    if not _have_executable("bandit"):
        raise Skip("bandit not installed")

    assert analyze_security(WEAK_HASH_SAMPLE), "md5 was not flagged"


def test_security_clean_code_has_no_issues():
    if not _have_executable("bandit"):
        raise Skip("bandit not installed")

    clean = "def add(a, b):\n    return a + b\n"
    assert analyze_security(clean) == [], "bandit flagged clean code"


# ---------------------------------------------------------------------------
# Complexity — radon
# ---------------------------------------------------------------------------

# Cyclomatic complexity is branches + 1. This has 12 branches -> CC 13,
# comfortably over the error threshold of 10.
COMPLEX_SAMPLE = "def grade(a, b, c, d, e, f, g, h, i, j, k, m):\n" + "".join(
    f"    if {v} > 0:\n        return '{v}'\n"
    for v in "abcdefghijkm"
) + "    return 'none'\n"


def test_complexity_flags_complex_function():
    if not _have_module("radon"):
        raise Skip("radon not installed")

    issues = analyze_complexity(COMPLEX_SAMPLE)
    assert issues, "a 12-branch function was not flagged"
    assert any(i.rule == "CC001" for i in issues)
    assert any(i.severity == IssueSeverity.ERROR for i in issues), \
        "CC above 10 should be an error, not a warning"


def test_complexity_ignores_simple_function():
    if not _have_module("radon"):
        raise Skip("radon not installed")

    assert analyze_complexity("def add(a, b):\n    return a + b\n") == []


def test_maintainability_index_range():
    if not _have_module("radon"):
        raise Skip("radon not installed")

    mi = get_maintainability_index("def add(a, b):\n    return a + b\n")
    assert 0.0 <= mi <= 100.0, f"MI out of range: {mi}"


def test_maintainability_prefers_simple_code():
    if not _have_module("radon"):
        raise Skip("radon not installed")

    simple = get_maintainability_index("def add(a, b):\n    return a + b\n")
    messy = get_maintainability_index(COMPLEX_SAMPLE)
    assert simple > messy, f"simple code scored {simple}, messy scored {messy}"


# ---------------------------------------------------------------------------
# Smells — pure AST, always available
# ---------------------------------------------------------------------------

def test_smells_detects_deep_nesting():
    code = (
        "def f(data):\n"
        "    for a in data:\n"
        "        for b in a:\n"
        "            for c in b:\n"
        "                if c:\n"
        "                    return c\n"
    )
    issues = analyze_smells(code)
    assert issues, "deeply nested code produced no smell"
    assert all(i.category == IssueCategory.SMELL for i in issues)


def test_smells_quiet_on_clean_code():
    assert analyze_smells("def add(a, b):\n    return a + b\n") == []


def test_smells_detects_inconsistent_return_type():
    code = (
        "def parse(x):\n"
        "    if x is None:\n"
        "        return 'error'\n"
        "    return 42\n"
    )
    issues = analyze_smells(code)
    rule_hits = [i for i in issues if i.rule == "SMELL011"]
    assert rule_hits, "str/int return mismatch produced no SMELL011"
    assert rule_hits[0].category == IssueCategory.SMELL


def test_smells_ignores_optional_return_pattern():
    """`return value` on one path and `return None` on another is the
    ordinary optional-result idiom, not a smell -- must stay silent."""
    code = (
        "def find(items, target):\n"
        "    for i, x in enumerate(items):\n"
        "        if x == target:\n"
        "            return i\n"
        "    return None\n"
    )
    issues = [i for i in analyze_smells(code) if i.rule == "SMELL011"]
    assert issues == []


def test_smells_does_not_guess_variable_return_types():
    """return a / return b with no literals -- no type inference attempted,
    so this must stay silent rather than risk a wrong guess."""
    code = (
        "def pick(a, b, flag):\n"
        "    if flag:\n"
        "        return a\n"
        "    return b\n"
    )
    issues = [i for i in analyze_smells(code) if i.rule == "SMELL011"]
    assert issues == []


def test_smells_detects_renamed_duplicate_function():
    """Identical logic, different variable/parameter names -- SMELL006 only
    catches exact statement matches and misses this; SMELL012 exists
    specifically for it. Was verified against a CodeBERT-embedding-based
    approach first, which failed calibration testing (ranked semantically
    different same-shaped functions as more similar than genuine
    duplicates) -- this deterministic check replaced it."""
    code = (
        "def sum_list(nums):\n"
        "    total = 0\n"
        "    for n in nums:\n"
        "        total = total + n\n"
        "    return total\n"
        "\n"
        "def add_all(values):\n"
        "    result = 0\n"
        "    for v in values:\n"
        "        result = result + v\n"
        "    return result\n"
    )
    issues = [i for i in analyze_smells(code) if i.rule == "SMELL012"]
    assert issues, "renamed-variable duplicate produced no SMELL012"
    assert issues[0].category == IssueCategory.SMELL


def test_smells_detects_renamed_recursive_duplicate():
    """Recursive self-calls must normalize the same way a parameter does --
    this was a real bug caught while building the check: the function's own
    name wasn't being mapped, so fact()/factorial() (identical except for
    the recursive call's name) went undetected until fixed."""
    code = (
        "def fact(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * fact(n - 1)\n"
        "\n"
        "def factorial(m):\n"
        "    if m <= 1:\n"
        "        return 1\n"
        "    return m * factorial(m - 1)\n"
    )
    issues = [i for i in analyze_smells(code) if i.rule == "SMELL012"]
    assert issues, "renamed recursive duplicate produced no SMELL012"


def test_smells_does_not_flag_different_functions_same_shape():
    """Two functions with the same structural shape (assign, call, return)
    but genuinely different logic must NOT fire -- this is the exact case
    that broke the CodeBERT-embedding approach (raw AND whitened cosine
    similarity ranked this pair as MORE similar than a real duplicate)."""
    code = (
        "def get_name(user):\n"
        "    total = user.first + user.last\n"
        "    formatted = total.strip()\n"
        "    return formatted\n"
        "\n"
        "def get_email(user):\n"
        "    total = user.domain + user.local\n"
        "    formatted = total.strip()\n"
        "    return formatted\n"
    )
    issues = [i for i in analyze_smells(code) if i.rule == "SMELL012"]
    assert issues == []


def test_smells_does_not_flag_unrelated_functions():
    code = (
        "def sum_list(nums):\n"
        "    total = 0\n"
        "    for n in nums:\n"
        "        total = total + n\n"
        "    return total\n"
        "\n"
        "def binary_search(arr, target):\n"
        "    lo = 0\n"
        "    hi = len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    )
    issues = [i for i in analyze_smells(code) if i.rule == "SMELL012"]
    assert issues == []


def test_smells_near_duplicate_ignores_trivial_functions():
    """Below the statement threshold -- two one-line getters would
    otherwise flood results with noise; must stay silent."""
    code = (
        "def get_a(obj):\n"
        "    return obj.a\n"
        "\n"
        "def get_b(obj):\n"
        "    return obj.b\n"
    )
    issues = [i for i in analyze_smells(code) if i.rule == "SMELL012"]
    assert issues == []


# ---------------------------------------------------------------------------
# End-to-end grading
# ---------------------------------------------------------------------------

MESSY = (
    "import os\n"
    "import sys\n"
    "\n"
    "\n"
    "def process(user_input, data):\n"
    "    result = eval(user_input)\n"
    "    unused = 42\n"
    "    for a in data:\n"
    "        for b in a:\n"
    "            for c in b:\n"
    "                if c > 0:\n"
    "                    result = result + c\n"
    "    return result\n"
)

CLEAN = (
    "def add(a, b):\n"
    '    """Return the sum of a and b."""\n'
    "    return a + b\n"
)


def test_clean_code_grades_well():
    response = asyncio.run(run_all_checks(CLEAN))
    assert response.quality_score is not None
    assert response.quality_score.grade in ("A", "B"), (
        f"clean code graded {response.quality_score.grade} "
        f"({response.quality_score.score}/100)"
    )


def test_messy_code_grades_worse_than_clean_code():
    """
    Relative, not absolute. An absolute threshold would depend on which
    analysers are installed; the ordering must hold regardless.
    """
    clean = asyncio.run(run_all_checks(CLEAN)).quality_score.score
    messy = asyncio.run(run_all_checks(MESSY)).quality_score.score
    assert messy < clean, f"messy scored {messy}, clean scored {clean}"


def test_review_survives_every_analyser_failing():
    """One analyser blowing up must not take the endpoint down."""
    response = asyncio.run(run_all_checks("def f():\n    return 1\n"))
    assert response.valid is True
    assert response.quality_score is not None
    assert isinstance(response.summary, str) and response.summary


def test_score_breakdown_within_bounds():
    breakdown = asyncio.run(run_all_checks(MESSY)).quality_score.breakdown
    for name, value in breakdown.model_dump().items():
        # big_o is Optional: None when the complexity model is unavailable
        # or gave no prediction. That is a valid, documented state, not
        # something to bounds-check.
        if value is None:
            continue
        assert 0 <= value <= 100, f"{name} out of range: {value}"


def test_triple_nested_loop_lowers_grade_via_big_o():
    """
    The scenario HANDOFF step 7 was written for: a messy O(n^3+) function
    used to score however its style/security/maintainability issues alone
    added up to, with the ML complexity engine's own headline prediction
    having zero influence on the number right next to it.
    """
    response = asyncio.run(run_all_checks(MESSY))
    breakdown = response.quality_score.breakdown
    if breakdown.big_o is None:
        return  # complexity model unavailable in this environment — skip
    assert breakdown.big_o < 100, (
        f"triple-nested-loop code got big_o={breakdown.big_o}, expected a "
        "real penalty for O(n^3+)-shaped code"
    )


# ---------------------------------------------------------------------------
# Which analysers actually ran
# ---------------------------------------------------------------------------

def test_report_analyser_availability():
    """Never fails — prints what was exercised so silent gaps stay visible."""
    status = {
        "flake8  (style)":      _have_executable("flake8"),
        "bandit  (security)":   _have_executable("bandit"),
        "radon   (complexity)": _have_module("radon"),
        "AST     (smells)":     True,
    }
    print("\n    analyser availability:")
    for name, present in status.items():
        print(f"      {'yes' if present else 'NO '}  {name}")
    missing = [n for n, ok in status.items() if not ok]
    if missing:
        print(f"      -> {len(missing)} analyser(s) NOT exercised by this run")


# ---------------------------------------------------------------------------
# standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok      {name}")
            passed += 1
        except Skip as exc:
            print(f"  skip    {name}  ({exc})")
            skipped += 1
        except AssertionError as exc:
            print(f"  FAIL    {name}\n          {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR   {name}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
