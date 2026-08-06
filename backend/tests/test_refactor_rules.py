"""
tests/test_refactor_rules.py — Refactor rule correctness.

Two kinds of test here, and the second is the important one.

1. FIRES / DOES-NOT-FIRE pairs
   For every rule: one case it must transform, and a lookalike case it must
   leave alone. A rule that only has positive tests will happily fire on
   things it should not.

2. SEMANTIC EQUIVALENCE (test_refactor_preserves_behaviour)
   Executes the original and the refactored code with real inputs and
   compares the results. A refactoring tool that changes what code does is
   worse than no tool at all, and structural assertions cannot catch that —
   `[len(result) for x in items]` looks perfectly reasonable in a diff and
   raises NameError when run.

   This test caught three real bugs on its first run:
     - accumulator read inside the appended expression  -> NameError
     - loop variable used after the loop                -> NameError
     - loop target shadowing the accumulator            -> wrong result

Runs under pytest, or standalone with `python tests/test_refactor_rules.py`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import OptimizationLevel          # noqa: E402
from app.refactor.orchestrator import run_all_rules  # noqa: E402
from app.refactor.rules import loop_opts          # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def refactor(code: str, level=OptimizationLevel.MEDIUM):
    result = run_all_rules(code, level=level)
    return result.refactored_code, [r.rule for r in result.applied_rules if r.applied]


def rule_fires(code: str, rule: str, level=OptimizationLevel.MEDIUM) -> bool:
    _, rules = refactor(code, level)
    return rule in rules


def apply_loop_opts(code: str):
    tree, changes = loop_opts.apply(ast.parse(code))
    return ast.unparse(tree), [c.rule for c in changes if c.applied]


# ---------------------------------------------------------------------------
# 1. Fires / does-not-fire pairs
# ---------------------------------------------------------------------------

def test_loop_to_comprehension_fires():
    code = (
        "def f(items):\n"
        "    result = []\n"
        "    for x in items:\n"
        "        result.append(x * 2)\n"
        "    return result\n"
    )
    out, rules = apply_loop_opts(code)
    assert "loop_to_comprehension" in rules
    assert "[x * 2 for x in items]" in out


def test_loop_to_comprehension_declines_self_reference():
    """The accumulator is not in scope inside a comprehension — must decline."""
    code = (
        "def f(items):\n"
        "    result = []\n"
        "    for x in items:\n"
        "        result.append(len(result))\n"
        "    return result\n"
    )
    _, rules = apply_loop_opts(code)
    assert "loop_to_comprehension" not in rules


def test_loop_to_comprehension_declines_live_loop_variable():
    """A for-loop leaks its variable; a comprehension does not."""
    code = (
        "def f(items):\n"
        "    result = []\n"
        "    for x in items:\n"
        "        result.append(x * 2)\n"
        "    print(x)\n"
        "    return result\n"
    )
    _, rules = apply_loop_opts(code)
    assert "loop_to_comprehension" not in rules


def test_loop_to_comprehension_declines_shadowed_target():
    code = (
        "def f(items):\n"
        "    result = []\n"
        "    for result in items:\n"
        "        result.append(1)\n"
        "    return items\n"
    )
    _, rules = apply_loop_opts(code)
    assert "loop_to_comprehension" not in rules


def test_loop_to_comprehension_declines_multi_statement_body():
    code = (
        "def f(items):\n"
        "    result = []\n"
        "    for x in items:\n"
        "        y = x * 2\n"
        "        result.append(y)\n"
        "    return result\n"
    )
    _, rules = apply_loop_opts(code)
    assert "loop_to_comprehension" not in rules


def test_redundant_list_constructor_fires():
    out, rules = apply_loop_opts("x = list([1, 2, 3])\n")
    assert "redundant_list_constructor" in rules
    assert "list(" not in out


def test_redundant_list_constructor_declines_real_conversion():
    """list(some_iterable) is a real conversion, not redundant."""
    _, rules = apply_loop_opts("x = list(range(5))\ny = list(other)\n")
    assert "redundant_list_constructor" not in rules


def test_redundant_dict_constructor_fires():
    out, rules = apply_loop_opts("x = dict({'a': 1})\n")
    assert "redundant_dict_constructor" in rules
    assert "dict(" not in out


def test_redundant_dict_constructor_declines_kwargs_form():
    _, rules = apply_loop_opts("x = dict(a=1, b=2)\n")
    assert "redundant_dict_constructor" not in rules


def test_range_len_is_advice_not_a_change():
    """range(len(x)) is reported, but must not be counted as an applied change."""
    code = (
        "def f(items):\n"
        "    for i in range(len(items)):\n"
        "        print(items[i])\n"
    )
    tree, changes = loop_opts.apply(ast.parse(code))
    advice = [c for c in changes if c.rule == "range_len_to_enumerate"]
    assert len(advice) == 1
    assert advice[0].applied is False, "range_len is advice, not an applied change"
    assert "range(len(items))" in ast.unparse(tree), "code must be unmodified"


def test_range_len_declines_two_arg_range():
    code = "def f(items):\n    for i in range(1, len(items)):\n        print(i)\n"
    tree, changes = loop_opts.apply(ast.parse(code))
    assert not [c for c in changes if c.rule == "range_len_to_enumerate"]


def test_constant_folding_fires():
    assert rule_fires("def f():\n    x = 2 + 3 * 4\n    return x\n", "constant_fold")


def test_dead_code_after_return_fires():
    code = "def f():\n    return 1\n    print('unreachable')\n"
    out, rules = refactor(code)
    assert "unreachable" not in out


def test_clean_code_is_returned_byte_identical():
    """The most important regression test: do no harm."""
    for code in [
        "def add(a, b):\n    return a + b\n",
        "def greet(name):\n    return f'hello {name}'\n",
        "class Point:\n\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n",
    ]:
        out, rules = refactor(code)
        assert out.strip() == code.strip(), f"clean code was modified:\n{out}"
        assert rules == [], f"rules fired on clean code: {rules}"


# ---------------------------------------------------------------------------
# simplify_conditions
# ---------------------------------------------------------------------------

def test_bool_comparison_not_rewritten_when_type_unknown():
    """
    `if x == True:` -> `if x:` changes behaviour for non-bool values
    (2 == True is False, but `if 2` is truthy). Must be advice only.
    """
    code = "def f(x):\n    if x == True:\n        return 'yes'\n    return 'no'\n"
    result = run_all_rules(code, OptimizationLevel.MEDIUM)
    assert "x == True" in result.refactored_code, "unsafe rewrite was applied"
    advice = [r for r in result.applied_rules if not r.applied]
    assert advice, "the issue should still be reported as advice"


def test_bool_comparison_rewritten_when_provably_bool():
    """A comparison result IS a bool, so the rewrite is safe there."""
    code = "def f(a, b):\n    if (a < b) == True:\n        return 1\n    return 0\n"
    result = run_all_rules(code, OptimizationLevel.MEDIUM)
    assert "== True" not in result.refactored_code
    assert any(r.applied and r.rule == "simplify_conditions" for r in result.applied_rules)


def test_redundant_else_after_return_removed():
    code = (
        "def f(x):\n"
        "    if x > 0:\n"
        "        return 'pos'\n"
        "    else:\n"
        "        return 'neg'\n"
    )
    out, rules = refactor(code)
    assert "else" not in out
    assert "simplify_conditions" in rules


def test_bool_comparison_semantics_preserved():
    """Exhaustive check across types that expose the == True / == False trap."""
    cases = [
        "def f(x):\n    if x == True:\n        return 'a'\n    return 'b'\n",
        "def f(x):\n    if x == False:\n        return 'a'\n    return 'b'\n",
        "def f(x):\n    if x != True:\n        return 'a'\n    return 'b'\n",
        "def f(x):\n    if x != False:\n        return 'a'\n    return 'b'\n",
        "def f(x):\n    if x is True:\n        return 'a'\n    return 'b'\n",
    ]
    values = [True, False, 1, 0, 2, -1, [], [1], "", "a", None, 0.0, 1.0]
    failures = []
    for code in cases:
        out, _ = refactor(code)
        ns_o, ns_r = {}, {}
        exec(code, ns_o)
        exec(out, ns_r)
        for v in values:
            if ns_o["f"](v) != ns_r["f"](v):
                failures.append(f"{code.splitlines()[1].strip()} with x={v!r}")
    assert not failures, "behaviour changed for: " + ", ".join(failures)


# ---------------------------------------------------------------------------
# per-function complexity
# ---------------------------------------------------------------------------

def _predict(code):
    from app.ml.complexity_predictor import predict_complexity
    return predict_complexity(code, code)


MIXED_FILE = (
    "def find_duplicates(nums):\n"
    "    duplicates = []\n"
    "    for i in range(len(nums)):\n"
    "        for j in range(i + 1, len(nums)):\n"
    "            if nums[i] == nums[j]:\n"
    "                duplicates.append(nums[i])\n"
    "    return duplicates\n"
    "\n"
    "def calculate_stats(data):\n"
    "    total = 0\n"
    "    for item in data:\n"
    "        total = total + item\n"
    "    return total / len(data)\n"
)


def test_each_function_is_analysed_separately():
    prediction = _predict(MIXED_FILE)
    if prediction is None:
        return  # scikit-learn unavailable
    names = {f.name for f in prediction.functions}
    assert names == {"find_duplicates", "calculate_stats"}, names
    by_name = {f.name: f for f in prediction.functions}
    assert by_name["find_duplicates"].complexity == "O(n\u00b2)"
    assert by_name["calculate_stats"].complexity == "O(n)"


def test_slowest_function_drives_the_headline():
    prediction = _predict(MIXED_FILE)
    if prediction is None:
        return
    assert prediction.before == "O(n\u00b2)"
    dominant = [f for f in prediction.functions if f.is_dominant]
    assert len(dominant) == 1
    assert dominant[0].name == "find_duplicates"


def test_mixed_file_keeps_high_confidence():
    """
    Analysing a mixed file as one blob produced a blended ~52% prediction.
    Per-function analysis should report the dominant function's own
    confidence, which is high.
    """
    prediction = _predict(MIXED_FILE)
    if prediction is None:
        return
    assert prediction.confidence >= 0.80, (
        f"confidence regressed to {prediction.confidence} — "
        f"per-function analysis may have stopped working"
    )


def test_single_function_file_still_works():
    prediction = _predict("def f(n):\n    for i in range(n):\n        print(i)\n")
    if prediction is None:
        return
    assert len(prediction.functions) == 1
    assert prediction.functions[0].is_dominant
    assert prediction.before == prediction.functions[0].complexity


def test_script_without_functions_falls_back_to_module():
    prediction = _predict("total = 0\nfor i in range(10):\n    total = total + i\n")
    if prediction is None:
        return
    assert prediction.functions == []
    assert prediction.before, "module-level fallback produced no prediction"


def test_class_methods_are_analysed():
    code = (
        "class Table:\n"
        "\n"
        "    def lookup(self, key):\n"
        "        return self.data[key]\n"
        "\n"
        "    def scan_pairs(self, rows):\n"
        "        found = []\n"
        "        for a in rows:\n"
        "            for b in rows:\n"
        "                found.append((a, b))\n"
        "        return found\n"
    )
    prediction = _predict(code)
    if prediction is None:
        return
    assert {f.name for f in prediction.functions} == {"lookup", "scan_pairs"}


# ---------------------------------------------------------------------------
# 2. Semantic equivalence — the test that actually matters
# ---------------------------------------------------------------------------

# (source, call expression) — the call must be deterministic and side-effect free
BEHAVIOUR_CASES = [
    ("def f(items):\n"
     "    result = []\n"
     "    for x in items:\n"
     "        result.append(x * 2)\n"
     "    return result\n", "f([1, 2, 3])"),

    ("def f(items):\n"
     "    result = []\n"
     "    for x in items:\n"
     "        result.append(len(result))\n"
     "    return result\n", "f([10, 20, 30])"),

    ("def f(items):\n"
     "    result = []\n"
     "    for x in items:\n"
     "        result.append(x)\n"
     "    return (result, x)\n", "f([1, 2, 3])"),

    ("def f(pairs):\n"
     "    out = []\n"
     "    for a, b in pairs:\n"
     "        out.append(a + b)\n"
     "    return out\n", "f([(1, 2), (3, 4)])"),

    ("def f(n):\n"
     "    squares = []\n"
     "    for i in range(n):\n"
     "        squares.append(i * i)\n"
     "    total = 0\n"
     "    for s in squares:\n"
     "        total = total + s\n"
     "    return (squares, total)\n", "f(5)"),

    ("def f(items):\n"
     "    seen = []\n"
     "    for x in items:\n"
     "        if x not in seen:\n"
     "            seen.append(x)\n"
     "    return seen\n", "f([1, 2, 2, 3, 1])"),

    ("def f(n):\n"
     "    x = 2 + 3 * 4\n"
     "    y = list([1, 2, 3])\n"
     "    z = dict({'a': 1})\n"
     "    return (x, y, z, n)\n", "f(7)"),

    ("def f(matrix):\n"
     "    flat = []\n"
     "    for row in matrix:\n"
     "        for cell in row:\n"
     "            flat.append(cell)\n"
     "    return flat\n", "f([[1, 2], [3, 4]])"),

    ("def f(items):\n"
     "    doubled = []\n"
     "    for x in items:\n"
     "        doubled.append(x * 2)\n"
     "    tripled = []\n"
     "    for x in items:\n"
     "        tripled.append(x * 3)\n"
     "    return (doubled, tripled)\n", "f([1, 2])"),

    ("def f(items):\n"
     "    out = []\n"
     "    for i in range(len(items)):\n"
     "        out.append(items[i] + i)\n"
     "    return out\n", "f([10, 20, 30])"),
]


def _evaluate(source: str, call: str):
    """Run source and return the result of `call`, or a marker for the exception."""
    namespace: dict = {}
    try:
        exec(compile(source, "<case>", "exec"), namespace)
        return ("value", repr(eval(call, namespace)))
    except Exception as exc:  # noqa: BLE001
        return ("raised", type(exc).__name__)


def test_refactor_preserves_behaviour():
    """
    Refactored code must produce exactly what the original produced —
    including raising the same exception type when the original raised.
    """
    failures = []

    for source, call in BEHAVIOUR_CASES:
        refactored, rules = refactor(source)

        # never emit code that will not parse
        try:
            ast.parse(refactored)
        except SyntaxError as exc:
            failures.append(f"{call}: refactored output does not parse — {exc}")
            continue

        before = _evaluate(source, call)
        after = _evaluate(refactored, call)

        if before != after:
            failures.append(
                f"{call}: behaviour changed\n"
                f"    rules applied : {rules}\n"
                f"    original      : {before}\n"
                f"    refactored    : {after}\n"
                f"    refactored code:\n{refactored}"
            )

    assert not failures, "\n\n".join(failures)


def test_refactor_output_always_parses_on_corpus():
    """
    Every sample in the ML corpus is real Python. Refactoring any of them must
    produce something that still parses — at every optimization level.
    """
    from app.ml.corpus import load_corpus

    sources, _ = load_corpus()
    failures = []

    for level in (OptimizationLevel.LOW, OptimizationLevel.MEDIUM, OptimizationLevel.HIGH):
        for source in sources:
            out, _ = refactor(source, level)
            try:
                ast.parse(out)
            except SyntaxError as exc:
                failures.append(f"[{level}] {source.splitlines()[0]}: {exc}")

    assert not failures, "refactor produced unparseable code:\n" + "\n".join(failures[:10])


# ---------------------------------------------------------------------------
# standalone runner (pytest is not always installed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {name}\n        {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
