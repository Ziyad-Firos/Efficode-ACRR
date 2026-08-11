"""
verify/inputs.py — plausible test-input generation for differential testing.

Infers a parameter's likely kind from how it's USED in the function body —
light static analysis, not full type inference. Subscripting/iteration
suggests a list, string methods suggest a string, dict-like access
suggests a dict, arithmetic suggests an int. This is a heuristic and will
get some functions wrong; type hints, when present, override it entirely
rather than compete with it (annotations are ground truth, structural
evidence is a guess).

Deliberately mirrors the kind of hand-rolled AST pattern-matching already
used throughout app/ml/features.py and app/review/smells.py — a vote-based
heuristic over usage sites, not real type inference.
"""

from __future__ import annotations

import ast
import random
import string as _string_module
from typing import Any, Dict, List, Optional

KIND_LIST = "list"
KIND_STRING = "string"
KIND_DICT = "dict"
KIND_INT = "int"
KIND_UNKNOWN = "unknown"

_STRING_METHODS = {"join", "split", "strip", "lstrip", "rstrip", "upper", "lower",
                    "replace", "startswith", "endswith", "format", "encode"}
# "pop" is deliberately NOT dict-only -- list.pop() and dict.pop(key) are
# both real. It used to live only in _DICT_METHODS, which made
# `tasks.pop(0)` outvote a genuine `list(tasks)` signal on the same
# parameter (dict methods are weighted +2, so one `.pop(0)` call beat the
# list evidence outright) -- found building ml/t5/pairs.py's
# pop0-to-deque template, where the parameter was misclassified as a dict
# and fed dict values instead of lists. "get"/"setdefault" ARE dict-only
# and stay exclusive; "pop" now votes for both, same weight either way,
# so other evidence (list(x), iteration, subscripting) breaks the tie.
_DICT_ONLY_METHODS = {"get", "setdefault", "keys", "values", "items", "update"}
_LIST_METHODS = {"append", "extend", "insert", "remove", "pop"}
_ARITH_OPS = (ast.Sub, ast.Add, ast.FloorDiv, ast.Div, ast.Mult, ast.Mod)


def infer_param_kind(func_node: ast.AST, param_name: str) -> str:
    """
    Classify how `param_name` is used inside func_node's body. Returns one
    of KIND_LIST / KIND_STRING / KIND_DICT / KIND_INT / KIND_UNKNOWN.

    A type annotation on the parameter, if present, wins outright — no
    voting needed, it's already the answer. Otherwise every usage site
    casts a vote and the highest count wins; ties favour list, since it's
    the most common single parameter shape in the DSA-style code this
    project's corpus and this verification harness both target.
    """
    hinted = _kind_from_annotation(func_node, param_name)
    if hinted is not None:
        return hinted

    votes = {KIND_LIST: 0, KIND_STRING: 0, KIND_DICT: 0, KIND_INT: 0}

    for node in ast.walk(func_node):
        if isinstance(node, ast.For) and _is_param_ref(node.iter, param_name):
            votes[KIND_LIST] += 1
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call) \
                and isinstance(node.iter.func, ast.Name) \
                and node.iter.func.id in ("enumerate", "reversed", "sorted") \
                and node.iter.args and _is_param_ref(node.iter.args[0], param_name):
            votes[KIND_LIST] += 1

        # sorted(param) / reversed(param) / list(param) / set(param), even
        # OUTSIDE a for-loop's iter position, e.g. `ordered = sorted(nums)`.
        # Found missing when a template hoisting a sort out of a loop --
        # `ordered = sorted(nums)` used as a plain assignment, never
        # `for x in sorted(nums)` -- left `nums` with zero votes at all,
        # fell through to KIND_UNKNOWN, and got fed an int. Not a bug in
        # the transformation being tested, a gap in this inference.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("sorted", "reversed", "list", "set", "tuple") \
                and node.args and _is_param_ref(node.args[0], param_name):
            votes[KIND_LIST] += 1

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len" \
                and node.args and _is_param_ref(node.args[0], param_name):
            votes[KIND_LIST] += 1

        if isinstance(node, ast.Subscript) and _is_param_ref(node.value, param_name):
            votes[KIND_LIST] += 1

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and _is_param_ref(node.func.value, param_name):
            attr = node.func.attr
            if attr in _STRING_METHODS:
                votes[KIND_STRING] += 2
            if attr in _DICT_ONLY_METHODS:
                votes[KIND_DICT] += 2
            if attr in _LIST_METHODS:
                votes[KIND_LIST] += 2
            if attr == "pop":
                # list.pop() and dict.pop(key) are both real -- "pop" votes
                # for both (list already gets +2 above via _LIST_METHODS),
                # weighted lower for dict so other evidence breaks the tie
                # instead of "pop" alone deciding it. Found this backwards:
                # pop used to sit only in dict methods at full weight, so
                # `tasks.pop(0)` outvoted a genuine `list(tasks)` call on
                # the same parameter and misclassified it as a dict.
                votes[KIND_DICT] += 1

        if isinstance(node, ast.Compare):
            membership_ops = any(isinstance(o, (ast.In, ast.NotIn)) for o in node.ops)
            if membership_ops:
                for comparator in node.comparators:
                    if _is_param_ref(comparator, param_name):
                        votes[KIND_LIST] += 1  # 'in' works for dict too; list is the more common case here
            else:
                # equality/ordering (==, <, >, <=, >=, !=) against a bare
                # variable, e.g. `nums[i] + nums[j] == target` — the single
                # most common shape for a lone scalar parameter, and one
                # that casts no OTHER vote on its own. Missing this meant
                # a parameter used only this way fell through to
                # KIND_UNKNOWN and got fed list/dict/string values in
                # verify_equivalent, which crashed the candidate for
                # reasons having nothing to do with the refactoring being
                # tested — a real bug, found by testing this module
                # against an actual two-parameter function.
                refs = [node.left] + node.comparators
                if any(_is_param_ref(r, param_name) for r in refs):
                    votes[KIND_INT] += 1

        if isinstance(node, ast.BinOp) and _is_param_ref(node.left, param_name) \
                and isinstance(node.op, _ARITH_OPS):
            votes[KIND_INT] += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" \
                and node.args and _is_param_ref(node.args[0], param_name):
            votes[KIND_INT] += 1

    if not any(votes.values()):
        return KIND_UNKNOWN
    return max(votes, key=lambda k: votes[k])


def _is_param_ref(node: ast.AST, param_name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == param_name


def _kind_from_annotation(func_node: ast.AST, param_name: str) -> Optional[str]:
    args = getattr(func_node, "args", None)
    if args is None:
        return None
    for arg in args.args:
        if arg.arg != param_name or arg.annotation is None:
            continue
        ann = arg.annotation
        name = None
        if isinstance(ann, ast.Name):
            name = ann.id
        elif isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
            name = ann.value.id  # List[int] -> "List"
        elif isinstance(ann, ast.Attribute):
            name = ann.attr
        if name in ("list", "List"):
            return KIND_LIST
        if name == "str":
            return KIND_STRING
        if name in ("dict", "Dict"):
            return KIND_DICT
        if name == "int":
            return KIND_INT
    return None


def generate_edge_case_values(kind: str, rng: random.Random) -> List[Any]:
    """
    A fixed, deliberately edge-case-heavy set of values for one parameter
    kind. Used for equivalence testing — see generate_sized_value for
    speed-measurement inputs, which need to scale rather than cover edges.
    """
    if kind == KIND_LIST:
        return [
            [],
            [rng.randint(-50, 50)],
            [rng.randint(-50, 50) for _ in range(10)],
            sorted(rng.randint(-50, 50) for _ in range(10)),
            sorted((rng.randint(-50, 50) for _ in range(10)), reverse=True),
            [5] * 8,  # all identical elements
            [rng.randint(-50, 50) for _ in range(200)],
        ]
    if kind == KIND_STRING:
        return [
            "",
            "a",
            "hello world",
            "a" * 500,
            "aaaaaaaaaa",
            "日本語テスト",  # non-ASCII -- a naive len()/index refactor can break on this
            "".join(rng.choice(_string_module.ascii_lowercase) for _ in range(30)),
        ]
    if kind == KIND_DICT:
        return [
            {},
            {"a": 1},
            {str(i): i for i in range(20)},
            {i: str(i) for i in range(20)},
        ]
    if kind == KIND_INT:
        return [0, 1, -1, 2, -2, 10, 100, -100, 10**6]
    # unknown -- a conservative spread across kinds. Some calls will error
    # out inside the sandbox; that's fine and useful, not a failure, as
    # long as BOTH versions error the same way (see differential.py).
    return [0, 1, [], [1, 2, 3], "", "x", {}, None]


def generate_sized_value(kind: str, size: int, rng: random.Random) -> Any:
    """A single value of `kind`, sized to `size` — for measure_speedup,
    which needs inputs that grow, not inputs that cover edges."""
    if kind == KIND_LIST:
        return [rng.randint(-1000, 1000) for _ in range(size)]
    if kind == KIND_STRING:
        return "".join(rng.choice(_string_module.ascii_lowercase) for _ in range(size))
    if kind == KIND_DICT:
        return {i: rng.randint(-1000, 1000) for i in range(size)}
    return size  # int / unknown fallback -- use the size itself as the value


def build_call_expr(func_name: str, args: List[Any]) -> str:
    """Render a call to func_name with these Python values as a source
    string, using repr() so the values embed as valid literal syntax."""
    return f"{func_name}({', '.join(repr(a) for a in args)})"


def build_test_cases(
    inputs_per_param: Dict[str, List[Any]],
    param_order: List[str],
    trials: int,
) -> List[List[Any]]:
    """
    Combine each parameter's own edge-case list into concrete test cases.

    NOT a full cartesian product — that explodes combinatorially past 2-3
    parameters (7 edge cases ^ 3 params = 343 cases before any cap).
    Instead cycles each parameter's list independently, so every
    individual edge case for every parameter gets exercised at least once,
    capped at `trials` total cases.
    """
    if not param_order:
        return [[]]
    max_len = max(len(inputs_per_param[p]) for p in param_order)
    count = min(max_len, trials) if trials else max_len
    return [
        [inputs_per_param[p][i % len(inputs_per_param[p])] for p in param_order]
        for i in range(count)
    ]
