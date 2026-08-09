"""
ml/features.py — Structural feature extraction for the complexity predictor.

Split out of complexity_predictor.py so that the SAME code path produces
features at training time and at prediction time. The previous design
hand-typed training vectors, which meant training data and the extractor
could silently disagree.

Feature vector (24 features)
----------------------------
  0  max_loop_depth              deepest nesting of loops AND comprehensions
  1  loop_count                  total loops (comprehension generators count)
  2  has_recursion               1 if any function calls itself
  3  nested_loop_count           loops appearing inside another loop
  4  branch_count                total if/elif nodes                [noise-ish]
  5  has_sorting_call            sorted() / .sort()
  6  has_subscript               indexing present                   [noise-ish]
  7  has_set_ops                 explicit set algebra
  8  function_count              number of function definitions
  9  total_statements            total statements                   [noise-ish]
 10  max_self_calls_per_path     THE recursion feature — see below
 11  has_memo_guard              `if k in cache: return cache[k]`
 12  loop_bound_is_quadratic     range(n*n) / range(n**2)
 13  recursion_halves_input      recursive call on a slice/half/partition
 14  linear_scan_in_loop         `x in <list>` or .index()/sum() inside a loop
 15  has_visited_guard           set/dict-backed membership guard
 16  comprehension_count         number of comprehensions/genexps
 17  while_loop_halves           while-loop that divides its range
 18  sort_inside_loop            a sort executed on every iteration
 19  string_concat_in_loop       `s = s + x` where s is a string — quadratic
 20  has_mutual_recursion        f calls g, g calls f — recursion with no self-call
 21  mutual_recursion_branches   cross-calls per path in that cycle (1 = linear,
                                 2+ = branching, i.e. exponential)
 22  bulk_collection_op          whole-collection work with no visible loop:
                                 set(a) & set(b), ''.join(x), x[:] — all O(n)
 23  has_convergence_flag_while  "while flag: flag = False; ... flag = True"
                                 idiom — NOT used to pick a label (see the
                                 function docstring: the same shape is O(n^2)
                                 for an early-exit sort and O(n^3) for
                                 Bellman-Ford). Only feeds explain()'s caveat.

Features 4, 6 and 9 carry little asymptotic signal. They are retained
deliberately: corpus.py generates variants that vary exactly these values
while holding the label fixed, which trains the forest to ignore them.

Why max_self_calls_per_path (feature 10)
----------------------------------------
Counting recursive calls in the whole function body cannot tell binary search
apart from merge sort — both contain two textual self-calls. But binary
search's two calls sit in mutually exclusive branches, so only ONE runs per
invocation, while merge sort's two both run. Taking the maximum over
execution paths (max across if/else, sum along a sequence) gives 1 for binary
search and 2 for merge sort, which is exactly the distinction between
O(log n) and O(n log n). The same measure gives 2 for naive Fibonacci, and
combining it with feature 13 separates "branches on a halved input"
(divide and conquer, n log n) from "branches on the full input"
(exponential).
"""

from __future__ import annotations

import ast
from typing import List, Optional, Set

FEATURE_COUNT = 24

FEATURE_NAMES = [
    "max_loop_depth", "loop_count", "has_recursion", "nested_loop_count",
    "branch_count", "has_sorting_call", "has_subscript", "has_set_ops",
    "function_count", "total_statements", "max_self_calls_per_path",
    "has_memo_guard", "loop_bound_is_quadratic", "recursion_halves_input",
    "linear_scan_in_loop", "has_visited_guard", "comprehension_count",
    "while_loop_halves", "sort_inside_loop", "string_concat_in_loop",
    "has_mutual_recursion", "mutual_recursion_branches",
    "bulk_collection_op", "has_convergence_flag_while",
]

_LOOPS = (ast.For, ast.While, ast.AsyncFor)
_COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

# Constructors whose results support O(1) membership tests.
_HASHED_CONSTRUCTORS = {"set", "dict", "frozenset", "Counter", "defaultdict", "OrderedDict"}


# ---------------------------------------------------------------------------
# Container inference — is `x in y` a hash lookup or a linear scan?
# ---------------------------------------------------------------------------

def _hashed_names(tree: ast.AST) -> Set[str]:
    """
    Names that are bound to a set/dict-like container somewhere in the module.

    This is what separates `if x in seen` (O(1), seen is a set) from
    `if x in results` (O(n), results is a list). Getting this wrong is the
    difference between calling two-sum-with-a-hashmap O(n) and O(n²).
    """
    names: Set[str] = set()

    for node in ast.walk(tree):
        # seen = set()  /  counts = {}  /  cache = defaultdict(int)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]

            hashed = False
            if isinstance(value, (ast.Dict, ast.Set, ast.DictComp, ast.SetComp)):
                hashed = True
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
                    and value.func.id in _HASHED_CONSTRUCTORS:
                hashed = True
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) \
                    and value.func.attr in _HASHED_CONSTRUCTORS:
                hashed = True

            # annotated: cache: dict = ...
            if isinstance(node, ast.AnnAssign) and node.annotation is not None:
                if _annotation_is_hashed(node.annotation):
                    hashed = True

            if hashed:
                for t in targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)

        # def f(memo: dict): / def f(seen: set[int]):
        if isinstance(node, ast.arg) and node.annotation is not None:
            if _annotation_is_hashed(node.annotation):
                names.add(node.arg)

        # def f(n, memo={}): — default argument is a dict/set literal
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = args.posonlyargs + args.args
            defaults = args.defaults
            if defaults:
                for arg, default in zip(positional[-len(defaults):], defaults):
                    if isinstance(default, (ast.Dict, ast.Set)):
                        names.add(arg.arg)
                    elif isinstance(default, ast.Call) and isinstance(default.func, ast.Name) \
                            and default.func.id in _HASHED_CONSTRUCTORS:
                        names.add(arg.arg)

    # A name used as `name[k] = v` with a non-integer key behaves like a dict.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    key = t.slice
                    if isinstance(key, ast.Name) or isinstance(key, ast.Constant) \
                            and isinstance(key.value, str):
                        names.add(t.value.id)

    return names


def _annotation_is_hashed(annotation: ast.AST) -> bool:
    try:
        text = ast.unparse(annotation).lower()
    except Exception:  # noqa: BLE001
        return False
    return any(k in text for k in ("set", "dict", "mapping", "counter"))


# ---------------------------------------------------------------------------
# Recursion shape
# ---------------------------------------------------------------------------

def _self_calls_in_expr(node: ast.AST, name: str) -> int:
    return _calls_in_expr(node, {name})


def _calls_in_expr(node: ast.AST, names: Set[str]) -> int:
    return sum(
        1 for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names
    )


def max_self_calls_per_path(fn: ast.AST, name: Optional[str] = None) -> int:
    """
    Maximum number of self-calls that execute on any single path through fn.

    Sequential statements add; if/else branches take the maximum; loop bodies
    count once (their repetition is already captured by the loop features).

    binary search -> 1   (two calls, mutually exclusive branches)
    merge sort    -> 2   (two calls, both execute)
    fib naive     -> 2   (two calls in one expression)
    """
    name = name or getattr(fn, "name", "")
    if not name:
        return 0
    return max_calls_per_path(fn, {name})


def max_calls_per_path(fn: ast.AST, names: Set[str]) -> int:
    """
    Maximum calls to any of `names` along a single path through fn.

    Same path arithmetic as max_self_calls_per_path — sequential statements
    add, branches take the maximum, an early return ends the path — but over
    an arbitrary set of callees. That generalisation is what lets mutual
    recursion be measured: the cycle members are the target set instead of
    the function's own name.
    """
    if not names:
        return 0

    def is_terminal(body: List[ast.stmt]) -> bool:
        """True if the block always exits — nothing after it runs."""
        return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise))

    def walk_body(body: List[ast.stmt]) -> int:
        """
        Maximum self-calls along any path through this statement list.

        Guard clauses matter here. In

            if node is None:  return None
            if key < node.key: return search(node.left, key)
            return search(node.right, key)

        there are two textual self-calls but only ONE ever runs, because the
        third statement is only reached when the second did not return.
        Summing gives 2 (the exponential signature); splitting at the early
        return gives 1 (the logarithmic one). Without this, binary search in
        a BST was indistinguishable from naive Fibonacci.
        """
        if not body:
            return 0

        head, rest = body[0], body[1:]

        if isinstance(head, ast.If):
            test_calls = _calls_in_expr(head.test, names)
            then_branch = walk_body(head.body)
            else_branch = walk_body(head.orelse)

            if is_terminal(head.body) and not head.orelse:
                # either we take the guard and stop, or we skip it and continue
                return test_calls + max(then_branch, walk_body(rest))
            if is_terminal(head.body) and is_terminal(head.orelse):
                return test_calls + max(then_branch, else_branch)
            return test_calls + max(then_branch, else_branch) + walk_body(rest)

        if isinstance(head, (ast.Return, ast.Raise)):
            return _calls_in_expr(head, names)   # path ends here

        if isinstance(head, ast.Try):
            return max(walk_body(head.body), walk_body(head.orelse)) + walk_body(rest)

        if isinstance(head, _LOOPS):
            # a loop body runs once per iteration; repetition is captured by
            # the loop features, so count it once
            return max(walk_body(head.body), walk_body(head.orelse)) + walk_body(rest)

        if isinstance(head, ast.With):
            return walk_body(head.body) + walk_body(rest)

        if isinstance(head, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return walk_body(rest)   # nested def is not executed inline

        return _calls_in_expr(head, names) + walk_body(rest)

    return walk_body(list(getattr(fn, "body", [])))


def _recursion_halves_input(fn: ast.AST) -> bool:
    """
    True when a recursive call receives a strictly smaller *partition* of the
    input rather than the whole thing: a slice (arr[:mid]), a floor-divided
    bound, or a name built by comprehending over the input.

    This is what makes merge sort and quicksort O(n log n) instead of
    exponential, despite both having two self-calls per path.
    """
    name = getattr(fn, "name", "")
    if not name:
        return False

    # names assigned from a slice, a floordiv, or a filtering comprehension
    derived: Set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            value = node.value
            for sub in ast.walk(value):
                if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice):
                    derived.add(target)
                if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.FloorDiv):
                    derived.add(target)
            if isinstance(value, _COMPS):
                derived.add(target)

    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == name):
            continue
        for arg in node.args:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice):
                    return True
                if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.FloorDiv):
                    return True
                if isinstance(sub, ast.Name) and sub.id in derived:
                    return True
                if isinstance(sub, _COMPS):
                    return True
    return False


def _has_memo_guard(fn: ast.AST) -> bool:
    """Detect `if key in cache: return cache[key]` near the top of a function."""
    for stmt in list(getattr(fn, "body", []))[:4]:
        if not isinstance(stmt, ast.If):
            continue
        test = stmt.test
        if not (isinstance(test, ast.Compare) and any(isinstance(o, ast.In) for o in test.ops)):
            continue
        if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Return):
            return True
    return False


def _loop_bound_is_quadratic(fn: ast.AST) -> bool:
    """Detect range(n * n) / range(n ** 2) — quadratic work at nesting depth 1."""
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "range"):
            continue
        for arg in node.args:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.BinOp) and isinstance(sub.op, (ast.Mult, ast.Pow)):
                    return True
    return False


# ---------------------------------------------------------------------------
# Loop / membership structure
# ---------------------------------------------------------------------------

def _loop_structure(tree: ast.AST):
    """
    Walk the tree tracking nesting depth, counting comprehensions as loops.

    A list comprehension is a loop. The old extractor ignored them entirely,
    which is why quicksort (whose partitioning is three comprehensions) looked
    like straight-line code.
    """
    max_depth = 0
    loop_count = 0
    nested = 0

    def walk(node: ast.AST, depth: int) -> None:
        nonlocal max_depth, loop_count, nested
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _LOOPS):
                loop_count += 1
                d = depth + 1
                max_depth = max(max_depth, d)
                if depth >= 1:
                    nested += 1
                walk(child, d)
            elif isinstance(child, _COMPS):
                levels = max(1, len(child.generators))
                loop_count += levels
                d = depth + levels
                max_depth = max(max_depth, d)
                if depth >= 1:
                    nested += levels
                elif levels > 1:
                    nested += levels - 1
                walk(child, d)
            else:
                walk(child, depth)

    walk(tree, 0)
    return max_depth, loop_count, nested


# Methods and builtins that cost O(n) in the size of their collection.
_LINEAR_METHODS = {"index", "count", "remove"}
_LINEAR_BUILTINS = {"sum", "min", "max", "any", "all", "reversed", "list", "tuple"}


def _loop_body_nodes(loop: ast.AST):
    """
    Children of a loop that actually execute per iteration.

    The iterable of a `for` and the test of a `while` are evaluated in the
    ENCLOSING scope, not once per iteration — so `for x in sorted(nums)` is
    not a sort inside a loop. Treating them as inner nodes would produce
    false quadratics on very ordinary code.
    """
    if isinstance(loop, (ast.For, ast.AsyncFor)):
        return list(loop.body) + list(loop.orelse)
    if isinstance(loop, ast.While):
        return list(loop.body) + list(loop.orelse)
    return list(ast.iter_child_nodes(loop))


def _is_linear_call(node: ast.AST, hashed: Set[str]) -> bool:
    """A call that scans a whole collection: items.index(x), sum(values), ..."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute):
        return node.func.attr in _LINEAR_METHODS
    if isinstance(node.func, ast.Name) and node.func.id in _LINEAR_BUILTINS:
        # max(a, b) on two scalars is O(1); max(values) on a collection is O(n).
        return len(node.args) == 1 and isinstance(node.args[0], ast.Name)
    return False


def _linear_scan_in_loop(tree: ast.AST, hashed: Set[str]) -> bool:
    """
    An O(n) operation evaluated inside a loop — the classic hidden quadratic.

    Two shapes, both common in hand-written DSA code:
      `if x in results:`      membership test against a list
      `items.index(target)`   a linear method call per iteration
    """
    found = False

    def scan_expr(node: ast.AST) -> None:
        """Look for linear operations anywhere in an expression."""
        nonlocal found
        for n in ast.walk(node):
            if isinstance(n, ast.Compare) and any(
                isinstance(o, (ast.In, ast.NotIn)) for o in n.ops
            ):
                for comparator in n.comparators:
                    if isinstance(comparator, ast.Name) and comparator.id not in hashed:
                        found = True
                    elif isinstance(comparator, (ast.List, ast.Tuple)):
                        found = True
            if _is_linear_call(n, hashed):
                found = True

    def walk(node: ast.AST, in_loop: bool) -> None:
        if isinstance(node, _LOOPS):
            # header evaluated outside the loop; body evaluated inside it
            for header in ([node.iter] if isinstance(node, (ast.For, ast.AsyncFor))
                           else [node.test]):
                walk(header, in_loop)
            for child in _loop_body_nodes(node):
                walk(child, True)
            return

        if isinstance(node, _COMPS):
            for gen in node.generators:
                walk(gen.iter, in_loop)
                for cond in gen.ifs:
                    walk(cond, True)
            for part in ("elt", "key", "value"):
                sub = getattr(node, part, None)
                if sub is not None:
                    walk(sub, True)
            return

        if in_loop:
            scan_expr(node)

        for child in ast.iter_child_nodes(node):
            walk(child, in_loop)

    walk(tree, False)
    return found


def _has_visited_guard(tree: ast.AST, hashed: Set[str]) -> bool:
    """
    A membership test against a hashed container inside a loop — the shape of
    BFS/DFS `if neighbour not in visited`. Structurally a graph traversal looks
    like a nested loop, but the guard makes each node process once, so it is
    linear, not quadratic. Without this feature BFS is indistinguishable from
    bubble sort.
    """
    found = False

    def walk(node: ast.AST, in_loop: bool) -> None:
        nonlocal found
        for child in ast.iter_child_nodes(node):
            child_in_loop = in_loop or isinstance(child, _LOOPS + _COMPS)
            if in_loop and isinstance(child, ast.Compare) \
                    and any(isinstance(o, (ast.In, ast.NotIn)) for o in child.ops):
                for comparator in child.comparators:
                    if isinstance(comparator, ast.Name) and comparator.id in hashed:
                        found = True
            walk(child, child_in_loop)

    walk(tree, False)
    return found


# Operations that shrink a value geometrically each iteration.
# RShift matters: `n = n >> 1` is the bit-twiddling spelling of `n = n // 2`,
# and missing it made count_set_bits look identical to a linked-list walk.
_HALVING_OPS = (ast.FloorDiv, ast.Div, ast.Mod, ast.RShift, ast.LShift)


def _names_referenced(node: ast.AST) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _while_loop_halves(tree: ast.AST) -> bool:
    """A while-loop whose OWN TEST is actually bounded by a variable the body
    divides or shifts — binary search, digit counting, bit counting, Euclid's
    algorithm.

    It is not enough for a halving op to appear anywhere inside the loop body:
    `while changed: ... fake = fake // 2 ...` is not geometrically shrinking,
    it just happens to contain unrelated division. That false positive was
    verified against a real adversarial sample (a while-loop guarded by a
    boolean flag, with a pointless `x // 2` several statements deep) — it
    produced a confident but hallucinated "halves its range" explanation.

    Requiring the halved value to share a name with something in the loop's
    own `test` catches every real case in the corpus (binary search halves
    via `mid = (lo + hi) // 2` where lo/hi ARE the test; digit/bit counting
    and Euclid's algorithm halve the exact variable the test reads) while
    rejecting arithmetic that has nothing to do with why the loop terminates.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        test_names = _names_referenced(node.test)
        if not test_names:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, _HALVING_OPS):
                if _names_referenced(sub) & test_names:
                    return True
            elif isinstance(sub, ast.AugAssign) and isinstance(sub.op, _HALVING_OPS):
                target_names = _names_referenced(sub.target)
                value_names = _names_referenced(sub.value)
                if (target_names | value_names) & test_names:
                    return True
    return False


def _flag_name(test: ast.AST) -> Optional[str]:
    """If `test` is a boolean-flag check (`while x:`, `while x == True:`,
    `while x is True:`, `while x != False:`), return the flag's name."""
    if isinstance(test, ast.Name):
        return test.id
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
        left, op, right = test.left, test.ops[0], test.comparators[0]
        if isinstance(left, ast.Name) and isinstance(right, ast.Constant) \
                and isinstance(right.value, bool):
            if isinstance(op, (ast.Eq, ast.Is)) and right.value is True:
                return left.id
            if isinstance(op, (ast.NotEq, ast.IsNot)) and right.value is False:
                return left.id
    return None


def _has_convergence_flag_while(tree: ast.AST) -> bool:
    """
    A while-loop guarded by a boolean flag the body resets to False up front
    and conditionally sets back to True — "keep going until nothing changed":
    early-exit bubble sort, Bellman-Ford relaxation, any fixed-point pass.

    This deliberately does NOT change the predicted complexity label. How
    many passes such a loop needs is a property of the algorithm's DATA, not
    its syntax. Verified on two real, equally realistic samples that share
    this exact shape: a comparison sort that always converges in one pass
    (true cost O(n^2)) and Bellman-Ford over an adjacency matrix, where each
    pass only propagates a shortest path one hop further, so convergence
    genuinely takes up to n passes (true cost O(n^3)) — confirmed empirically
    on both, passes growing with n on the second. No AST-level feature can
    tell these apart; it requires reasoning about what the comparison
    computes, not what it looks like. This feature exists so `explain()` can
    say so honestly instead of stating one of the two guesses with unearned
    confidence.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        flag = _flag_name(node.test)
        if flag is None or not node.body:
            continue
        first = node.body[0]
        resets = (
            isinstance(first, ast.Assign)
            and len(first.targets) == 1
            and isinstance(first.targets[0], ast.Name)
            and first.targets[0].id == flag
            and isinstance(first.value, ast.Constant)
            and first.value.value is False
        )
        if not resets:
            continue
        for sub in ast.walk(node):
            if sub is first:
                continue
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 \
                    and isinstance(sub.targets[0], ast.Name) \
                    and sub.targets[0].id == flag \
                    and isinstance(sub.value, ast.Constant) and sub.value.value is True:
                return True
    return False


# ---------------------------------------------------------------------------
# Costs the original 18 features could not see
# ---------------------------------------------------------------------------
# Each of these was added after an audit case the model got wrong. They are
# not hypothetical gaps — every one corresponds to a real miss.

_SORT_CALLS = {"sorted"}
_SORT_METHODS = {"sort"}


def _sort_inside_loop(tree: ast.AST) -> bool:
    """
    A sort executed once per iteration.

    `has_sorting_call` is only a flag: it fires the same whether the sort runs
    once or n times. That made `for g in groups: out.append(sorted(g))` look
    like a single O(n log n) sort instead of the n-sorts it actually is.
    """
    found = False

    def contains_sort(node: ast.AST) -> bool:
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name) and n.func.id in _SORT_CALLS:
                    return True
                if isinstance(n.func, ast.Attribute) and n.func.attr in _SORT_METHODS:
                    return True
        return False

    def walk(node: ast.AST, in_loop: bool) -> None:
        nonlocal found
        if isinstance(node, _LOOPS):
            header = node.iter if isinstance(node, (ast.For, ast.AsyncFor)) else node.test
            walk(header, in_loop)                      # header runs outside
            for child in list(node.body) + list(node.orelse):
                walk(child, True)
            return
        if isinstance(node, _COMPS):
            for gen in node.generators:
                walk(gen.iter, in_loop)
            for part in ("elt", "key", "value"):
                sub = getattr(node, part, None)
                if sub is not None:
                    walk(sub, True)
            return
        if in_loop and contains_sort(node):
            found = True
        for child in ast.iter_child_nodes(node):
            walk(child, in_loop)

    walk(tree, False)
    return found


def _string_names(tree: ast.AST) -> Set[str]:
    """Names that hold a string, inferred from their initialiser."""
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                names.add(node.targets[0].id)
            elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
                    and value.func.id == "str":
                names.add(node.targets[0].id)
            elif isinstance(value, ast.JoinedStr):
                names.add(node.targets[0].id)
        if isinstance(node, ast.arg) and node.annotation is not None:
            try:
                if "str" in ast.unparse(node.annotation).lower():
                    names.add(node.arg)
            except Exception:  # noqa: BLE001
                pass
    return names


def _string_concat_in_loop(tree: ast.AST) -> bool:
    """
    `out = out + piece` inside a loop, where out is a string.

    Python strings are immutable, so each concatenation copies the whole
    accumulated string — the loop is quadratic, not linear. The identical
    shape on a number (`total = total + x`) is genuinely O(1) per step, which
    is why this checks the inferred type rather than the syntax alone.
    """
    strings = _string_names(tree)
    if not strings:
        return False

    found = False

    def is_concat(node: ast.AST) -> bool:
        # out = out + piece
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in strings and isinstance(node.value, ast.BinOp) \
                    and isinstance(node.value.op, ast.Add):
                operands = {n.id for n in ast.walk(node.value)
                            if isinstance(n, ast.Name)}
                return name in operands
        # out += piece
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) \
                and isinstance(node.op, ast.Add):
            return node.target.id in strings
        return False

    def walk(node: ast.AST, in_loop: bool) -> None:
        nonlocal found
        for child in ast.iter_child_nodes(node):
            child_in_loop = in_loop or isinstance(child, _LOOPS + _COMPS)
            if in_loop and is_concat(child):
                found = True
            walk(child, child_in_loop)

    walk(tree, False)
    return found


def _has_mutual_recursion(tree: ast.AST) -> bool:
    """
    A cycle in the call graph that is not a self-call.

    `is_even(n)` calling `is_odd(n-1)` calling `is_even(n-2)` is recursion,
    but no function calls itself, so every self-call feature reads zero and
    the code looks like straight-line work.
    """
    graph: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            graph[node.name] = {
                n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
    if len(graph) < 2:
        return False

    # depth-first search for a cycle spanning at least two functions
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {name: WHITE for name in graph}

    def visit(name: str, depth: int) -> bool:
        colour[name] = GREY
        for callee in graph.get(name, ()):
            if callee not in graph or callee == name:
                continue                      # unknown call, or plain self-recursion
            if colour[callee] == GREY:
                return True                   # back edge across two functions
            if colour[callee] == WHITE and visit(callee, depth + 1):
                return True
        colour[name] = BLACK
        return False

    return bool(_mutual_recursion_cycle(tree))


def _mutual_recursion_cycle(tree: ast.AST) -> Set[str]:
    """Names of functions taking part in a non-self recursive cycle."""
    graph: dict = {}
    nodes: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes[node.name] = node
            graph[node.name] = {
                n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
    if len(graph) < 2:
        return set()

    members: Set[str] = set()
    for start in graph:
        # is `start` reachable from itself through at least one other function?
        stack = [(callee, 1) for callee in graph[start] if callee in graph and callee != start]
        seen = set()
        while stack:
            current, depth = stack.pop()
            if current == start and depth >= 1:
                members.add(start)
                break
            if current in seen:
                continue
            seen.add(current)
            for callee in graph.get(current, ()):
                if callee in graph:
                    stack.append((callee, depth + 1))
    return members


def _mutual_recursion_branches(tree: ast.AST) -> int:
    """
    Cross-calls per path inside a mutual-recursion cycle.

    1  -> linear (is_even calls is_odd once, which calls is_even once)
    2+ -> branching, so the call tree doubles at each level: exponential

    Without this, both shapes carry the same has_mutual_recursion flag and
    are indistinguishable — which produced contradictory training samples
    labelled O(n) and O(2^n) with identical feature vectors.
    """
    cycle = _mutual_recursion_cycle(tree)
    if not cycle:
        return 0

    worst = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in cycle:
            targets = cycle - {node.name}
            worst = max(worst, max_calls_per_path(node, targets))
    return worst


# Builtins and methods that touch every element of a collection. None of
# them is a loop in the AST, but all of them cost O(n) — which is why
# `set(a).intersection(set(b))` and `'-'.join(words)` were both predicted
# O(1): the extractor could see no loop, no recursion, and nothing else.
_BULK_BUILTINS = {
    "sum", "min", "max", "any", "all", "list", "tuple",
    "set", "frozenset", "reversed",
}
_BULK_METHODS = {
    "join", "copy", "extend", "intersection", "union", "difference",
    "symmetric_difference", "issubset", "issuperset", "count", "index",
    "remove", "split", "splitlines", "strip", "replace",
}
# Excluded because they are O(1): len, append, pop, add, get, keys,
# values, items (lazy views), enumerate/zip/range (lazy).
#
# sorted() and .sort() are also excluded — deliberately. They already have
# their own feature (has_sorting_call), and listing them here too made
# `bulk_collection_op` ambiguous between linear and n-log-n work: nine
# O(n log n) samples started reading as O(n). Keeping this feature to mean
# strictly LINEAR bulk work restores the separation.


def _bulk_collection_op(tree: ast.AST) -> bool:
    """
    A whole-collection operation somewhere in the code.

    Only meaningful in combination with the loop features: at depth 0 it
    signals linear work that has no loop to reveal it; inside a loop the
    cost is already captured by `linear_scan_in_loop`.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in _BULK_METHODS:
                return True
            if isinstance(node.func, ast.Name) and node.func.id in _BULK_BUILTINS:
                # min(a, b) on two scalars is O(1); min(values) is O(n).
                if len(node.args) == 1:
                    return True
        # a full slice copies the sequence
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            return True
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_features(code: str) -> Optional[List[int]]:
    """
    Extract the 24-element feature vector. Returns None if code does not parse.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return extract_features_from_tree(tree)


def extract_features_from_tree(tree: ast.AST) -> List[int]:
    hashed = _hashed_names(tree)

    branch_count = 0
    has_sorting = 0
    has_subscript = 0
    has_set_ops = 0
    function_count = 0
    total_statements = 0
    comprehension_count = 0

    self_calls = 0
    memo_guard = 0
    quad_bound = 0
    halves_input = 0

    func_names: Set[str] = set()
    called_names: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_names.add(node.name)
            function_count += 1

            calls_per_path = max_self_calls_per_path(node)
            self_calls = max(self_calls, calls_per_path)

            if calls_per_path > 0 and _has_memo_guard(node):
                memo_guard = 1
            if _loop_bound_is_quadratic(node):
                quad_bound = 1
            if calls_per_path > 0 and _recursion_halves_input(node):
                halves_input = 1

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
                if node.func.id == "sorted":
                    has_sorting = 1
            elif isinstance(node.func, ast.Attribute):
                # heapq.heappush/heappop cost log n each; used across n items
                # that is the same n log n shape as a sort. Without this,
                # a heap-based top-k looked like a plain linear scan.
                if node.func.attr in ("sort", "heappush", "heappop", "heapify",
                                      "heappushpop", "heapreplace",
                                      "nlargest", "nsmallest"):
                    has_sorting = 1
                if node.func.attr in ("intersection", "union", "difference",
                                      "issubset", "issuperset", "symmetric_difference"):
                    has_set_ops = 1

        if isinstance(node, ast.Subscript):
            has_subscript = 1
        if isinstance(node, ast.If):
            branch_count += 1
        if isinstance(node, ast.stmt):
            total_statements += 1
        if isinstance(node, _COMPS):
            comprehension_count += 1

    max_depth, loop_count, nested = _loop_structure(tree)
    has_recursion = int(bool(func_names & called_names))

    return [
        max_depth,                                      # 0
        loop_count,                                     # 1
        has_recursion,                                  # 2
        nested,                                         # 3
        branch_count,                                   # 4
        has_sorting,                                    # 5
        has_subscript,                                  # 6
        has_set_ops,                                    # 7
        function_count,                                 # 8
        total_statements,                               # 9
        self_calls,                                     # 10
        memo_guard,                                     # 11
        quad_bound,                                     # 12
        halves_input,                                   # 13
        int(_linear_scan_in_loop(tree, hashed)),        # 14
        int(_has_visited_guard(tree, hashed)),          # 15
        comprehension_count,                            # 16
        int(_while_loop_halves(tree)),                  # 17
        int(_sort_inside_loop(tree)),                   # 18
        int(_string_concat_in_loop(tree)),              # 19
        int(_has_mutual_recursion(tree)),               # 20
        _mutual_recursion_branches(tree),               # 21
        int(_bulk_collection_op(tree)),                 # 22
        int(_has_convergence_flag_while(tree)),          # 23
    ]
