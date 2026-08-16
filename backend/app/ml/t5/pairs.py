"""
ml/t5/pairs.py — hand-written (before, after) transformation pairs.

Why both sides are written by hand, never generated
-----------------------------------------------------
Do NOT generate the "after" side by running Groq (or any LLM). That is
distillation: a model fine-tuned on it would learn to imitate Groq,
including its mistakes, and could never exceed it — this project would
spend real effort reproducing a capability it already has. There would
also be no way to know if a pair was semantically wrong, because nothing
would have verified it. Every before/after pair here is written directly
as real Python, correct by construction, with the transformation label
exact because no model was consulted to produce either side.

Every pair is validated before being trusted
----------------------------------------------
See build_dataset() at the bottom: every generated (before, after) pair is
run through verify_equivalent() (app/verify/differential.py) and
measure_speedup(). A pair that fails either check is a bug in the
TEMPLATE, not a reason to drop the check — see the module docstring
convention already established by app/ml/corpus.py's contradiction
checking and app/ml/audit.py's adversarial stance for the same principle
applied to a different dataset.

Scope, stated honestly
------------------------
The plan calls for 15-20 transformation families x ~250 mutations each
(4,000-5,000 pairs). This implements 10 families — diverse, each a genuine
complexity-class change verifiable by measure_speedup, deliberately
avoiding transformations with inherent output ambiguity (see the
pair_sum_count family's docstring for why "find the first matching pair"
was rejected as a template shape after it produced a real bug in
tests/test_differential.py's own calibration). ~25 instances per family
(~250 pairs) keeps full validation — every single pair, no sampling —
under a few minutes. Scaling to the full 4,000-5,000 is a mechanical
extension of this same infrastructure (more families in the same shape,
more instances per family), not a new engineering problem; not done here
given the time this would add versus the plan's day budget.

The last 2 families (running_sum_to_cumulative, running_count_to_incremental)
were added later than the original 8, specifically to teach a concept a
real fine-tuning run showed was missing -- see
_gen_running_sum_to_cumulative's docstring for the full reasoning. Both
are TRAINED, not held out: the held-out set (build_dataset.py's
_holdout_families) is pinned explicitly to the original two
(minmax_in_loop_to_running, pop0_to_deque) specifically so results stay
comparable across training runs as more trained families are added —
letting the held-out set silently shift with FAMILIES' length would break
every existing "34% pass rate, gate 2 passes" comparison point.

Mutation strategy
--------------------
Each template samples fresh identifier names per call via random.Random,
so N calls to the same template produce N structurally-identical,
lexically-different pairs — same principle as app/ml/corpus.py's
_variants(), applied at generation time instead of as a post-processing
pass over a single fixed sample.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# (before_code, after_code, function_name, custom_test_values)
TemplateResult = Tuple[str, str, str, Dict[str, list]]


@dataclass
class TransformationFamily:
    name: str
    complexity_change: str
    generate: Callable[[random.Random], TemplateResult]


@dataclass
class TrainingPair:
    family: str
    complexity_change: str
    before: str
    after: str
    function_name: str
    custom_values: Dict[str, list] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def _gen_membership_to_set(rng: random.Random) -> TemplateResult:
    fn = rng.choice(["find_common", "filter_present", "select_matching", "keep_shared"])
    a = rng.choice(["items", "values", "data", "elements"])
    b = rng.choice(["target", "reference", "pool", "allowed"])
    acc = rng.choice(["result", "output", "matches", "found"])
    before = (
        f"def {fn}({a}, {b}):\n"
        f"    {acc} = []\n"
        f"    for item in {a}:\n"
        f"        if item in {b}:\n"
        f"            {acc}.append(item)\n"
        f"    return {acc}\n"
    )
    after = (
        f"def {fn}({a}, {b}):\n"
        f"    lookup = set({b})\n"
        f"    return [item for item in {a} if item in lookup]\n"
    )
    return before, after, fn, {}


def _gen_index_to_dict(rng: random.Random) -> TemplateResult:
    """
    O(n^2) -> O(n). Found a real bug building this: `.index()` always
    returns the FIRST matching index, but the obvious dict-comprehension
    rewrite `{val: idx for idx, val in enumerate(ref)}` keeps the LAST
    occurrence on a duplicate key (later entries overwrite earlier ones in
    a dict comprehension). Caught immediately by build_dataset.py's
    validation pass on a ref list of 8 identical values: original returned
    index 0 every time, the naive rewrite returned 7 every time. Fixed by
    only recording the first occurrence, mirroring _direct_returns-style
    "check the specific semantics, don't assume the obvious rewrite is
    equivalent" discipline used throughout this project.
    """
    fn = rng.choice(["positions_of", "find_indices", "lookup_positions"])
    items = rng.choice(["items", "queries", "keys"])
    ref = rng.choice(["catalog", "reference", "source"])
    acc = rng.choice(["result", "positions", "indices"])
    before = (
        f"def {fn}({items}, {ref}):\n"
        f"    {acc} = []\n"
        f"    for item in {items}:\n"
        f"        if item in {ref}:\n"
        f"            {acc}.append({ref}.index(item))\n"
        f"    return {acc}\n"
    )
    after = (
        f"def {fn}({items}, {ref}):\n"
        f"    position = {{}}\n"
        f"    for idx, val in enumerate({ref}):\n"
        f"        if val not in position:\n"
        f"            position[val] = idx\n"
        f"    {acc} = []\n"
        f"    for item in {items}:\n"
        f"        if item in position:\n"
        f"            {acc}.append(position[item])\n"
        f"    return {acc}\n"
    )
    return before, after, fn, {}


def _gen_string_concat_to_join(rng: random.Random) -> TemplateResult:
    """
    O(n^2) -> O(n) in the textbook sense (repeated `s = s + x` reallocates
    the whole string each time, since strings are immutable). Correctness
    (verify_equivalent) holds -- both sides are confirmed behaviourally
    identical. But measure_speedup does NOT reliably show a growing
    speedup trend for this one on CPython specifically: CPython has a
    real, documented runtime optimization (PyUnicode's in-place resize
    path) that special-cases exactly this `s = s + x` / `s += x` pattern
    when the left-hand string's refcount is 1, which it is in this simple
    loop. So the "obviously quadratic" textbook claim doesn't always show
    up as a measured wall-clock difference here. Not a bug in the
    transformation or the measurement -- a real fact about the difference
    between Big-O theory and CPython's actual implementation, worth
    knowing rather than hiding. Kept in the dataset with its textbook
    label; a fine-tuned model should still learn the (correct, real-world
    valuable) join-based rewrite even though this particular measurement
    doesn't confirm a wall-clock win at these sizes.
    """
    fn = rng.choice(["build_string", "concat_all", "join_items", "assemble_text"])
    items = rng.choice(["parts", "words", "tokens", "pieces"])
    acc = rng.choice(["result", "text", "output"])
    before = (
        f"def {fn}({items}):\n"
        f"    {acc} = \"\"\n"
        f"    for item in {items}:\n"
        f"        {acc} = {acc} + str(item)\n"
        f"    return {acc}\n"
    )
    after = (
        f"def {fn}({items}):\n"
        f"    return \"\".join(str(item) for item in {items})\n"
    )
    return before, after, fn, {}


def _gen_naive_fib_to_memo(rng: random.Random) -> TemplateResult:
    """
    O(2^n) -> O(n). The one family in this set with an implicit input-size
    constraint the generic KIND_INT edge cases don't know about — a
    naively-recursive call blows the sandbox timeout (or Python's own
    recursion limit) on the generic edge case of 10**6. custom_values
    (added to verify/differential.py specifically for this template)
    supplies small values instead; not a bug in the transformation, a
    property of this shape of function.
    """
    fn = rng.choice(["fib", "fibonacci", "fib_seq", "count_ways"])
    n = rng.choice(["n", "steps", "k"])
    before = (
        f"def {fn}({n}):\n"
        f"    if {n} <= 1:\n"
        f"        return {n}\n"
        f"    return {fn}({n} - 1) + {fn}({n} - 2)\n"
    )
    after = (
        "from functools import lru_cache\n\n"
        "@lru_cache(maxsize=None)\n"
        f"def {fn}({n}):\n"
        f"    if {n} <= 1:\n"
        f"        return {n}\n"
        f"    return {fn}({n} - 1) + {fn}({n} - 2)\n"
    )
    return before, after, fn, {n: [0, 1, 2, 3, 5, 8, 12, 16, 20]}


def _gen_pair_sum_count_to_hashmap(rng: random.Random) -> TemplateResult:
    """
    O(n^2) -> O(n). Counts pairs (i < j) summing to a target, rather than
    returning the FIRST matching pair's indices.

    This shape was chosen deliberately after a real failure: an earlier
    version of this exact idea (verify_equivalent's own calibration test,
    find_pair — return the first (i, j) found) turned out to be genuinely
    ambiguous whenever the input has duplicate values, because a hash-map
    rewrite naturally visits pairs in a different order than nested loops
    do, and "first" depends on that order. That produced a real, valid
    disagreement that had nothing to do with the transformation being
    wrong (see tests/test_differential.py's
    test_correct_refactoring_verifies_as_equivalent for the version that
    replaced it, and test_ai_client.py's confirmed instance of a live Groq
    suggestion hitting the identical bug). A COUNT has no such ambiguity —
    exactly one correct number exists regardless of visiting order — so it
    is the right choice for a training example, where an accidentally
    ambiguous "before" would poison the label.
    """
    fn = rng.choice(["count_pairs", "pairs_with_sum", "count_pair_sum"])
    nums = rng.choice(["nums", "values", "numbers"])
    target = rng.choice(["target", "goal", "total"])
    acc = rng.choice(["count", "total_pairs", "matches"])
    before = (
        f"def {fn}({nums}, {target}):\n"
        f"    {acc} = 0\n"
        f"    for i in range(len({nums})):\n"
        f"        for j in range(i + 1, len({nums})):\n"
        f"            if {nums}[i] + {nums}[j] == {target}:\n"
        f"                {acc} = {acc} + 1\n"
        f"    return {acc}\n"
    )
    after = (
        f"def {fn}({nums}, {target}):\n"
        f"    seen = {{}}\n"
        f"    {acc} = 0\n"
        f"    for num in {nums}:\n"
        f"        complement = {target} - num\n"
        f"        if complement in seen:\n"
        f"            {acc} = {acc} + seen[complement]\n"
        f"        seen[num] = seen.get(num, 0) + 1\n"
        f"    return {acc}\n"
    )
    return before, after, fn, {}


def _gen_sort_in_loop_to_once(rng: random.Random) -> TemplateResult:
    """
    O(k * n log n) -> O(n log n + k) -- resorting the same collection on
    every loop iteration when it never changes, instead of once outside.
    Correctness holds (verify_equivalent confirms it). measure_speedup
    cannot measure this one reliably: `queries` is used as a list index
    (`ordered[q]`), but generate_sized_value has no way to know that and
    fills it with generic large random ints, which are out of range for
    `ordered` at most sizes and raise IndexError identically in both
    versions -- a real, understood limitation of generic value generation
    for parameters with an implicit "valid index" constraint, not a
    correctness problem (verify_equivalent's smaller, saner edge-case
    values never hit this). Left in the dataset with its textbook label;
    fixing measure_speedup to be index-aware is future work, not done here.
    """
    fn = rng.choice(["ranked_queries", "sorted_lookup", "rank_at"])
    nums = rng.choice(["nums", "values", "scores"])
    queries = rng.choice(["queries", "indices", "positions"])
    acc = rng.choice(["result", "output", "answers"])
    before = (
        f"def {fn}({nums}, {queries}):\n"
        f"    {acc} = []\n"
        f"    for q in {queries}:\n"
        f"        ordered = sorted({nums})\n"
        f"        {acc}.append(ordered[q] if q < len(ordered) else None)\n"
        f"    return {acc}\n"
    )
    after = (
        f"def {fn}({nums}, {queries}):\n"
        f"    ordered = sorted({nums})\n"
        f"    {acc} = []\n"
        f"    for q in {queries}:\n"
        f"        {acc}.append(ordered[q] if q < len(ordered) else None)\n"
        f"    return {acc}\n"
    )
    return before, after, fn, {}


def _gen_minmax_in_loop_to_running(rng: random.Random) -> TemplateResult:
    """O(n^2) -> O(n) -- recomputing max()/min() over a growing slice at
    every index, instead of tracking a running max/min incrementally."""
    fn = rng.choice(["running_max_diff", "max_minus_min_each", "spread_at_each"])
    nums = rng.choice(["nums", "values", "readings"])
    acc = rng.choice(["result", "output", "diffs"])
    before = (
        f"def {fn}({nums}):\n"
        f"    {acc} = []\n"
        f"    for i in range(len({nums})):\n"
        f"        window = {nums}[:i + 1]\n"
        f"        {acc}.append(max(window) - min(window))\n"
        f"    return {acc}\n"
    )
    after = (
        f"def {fn}({nums}):\n"
        f"    {acc} = []\n"
        f"    current_max = None\n"
        f"    current_min = None\n"
        f"    for value in {nums}:\n"
        f"        if current_max is None or value > current_max:\n"
        f"            current_max = value\n"
        f"        if current_min is None or value < current_min:\n"
        f"            current_min = value\n"
        f"        {acc}.append(current_max - current_min)\n"
        f"    return {acc}\n"
    )
    return before, after, fn, {}


def _gen_running_sum_to_cumulative(rng: random.Random) -> TemplateResult:
    """
    O(n^2) -> O(n) -- recomputing sum() over a growing slice at every
    index, instead of tracking a running total incrementally.

    Added specifically to address a diagnosed weak spot, not part of the
    original 6-family training set: the first real go/no-go-eligible
    fine-tuning run (34% pass rate, gate 2 passing) generalized well to
    pop0_to_deque (15/25) but poorly to minmax_in_loop_to_running (2/25)
    -- the one held-out family requiring the "maintain a running value
    across a growing window instead of rescanning it" concept, which none
    of the 6 trained families taught explicitly. This template and
    _gen_running_count_to_incremental below teach that same general
    concept through two different concrete operations (sum, then a
    conditional count) WITHOUT duplicating minmax_in_loop_to_running's own
    shape -- the point is generalizing the underlying idea, not
    memorizing a near-copy of the held-out test itself, which would
    invalidate the held-out comparison rather than actually improve it.
    """
    fn = rng.choice(["running_total", "cumulative_sum", "prefix_sums"])
    nums = rng.choice(["nums", "values", "amounts"])
    acc = rng.choice(["result", "output", "totals"])
    before = (
        f"def {fn}({nums}):\n"
        f"    {acc} = []\n"
        f"    for i in range(len({nums})):\n"
        f"        window = {nums}[:i + 1]\n"
        f"        {acc}.append(sum(window))\n"
        f"    return {acc}\n"
    )
    after = (
        f"def {fn}({nums}):\n"
        f"    {acc} = []\n"
        f"    total = 0\n"
        f"    for value in {nums}:\n"
        f"        total = total + value\n"
        f"        {acc}.append(total)\n"
        f"    return {acc}\n"
    )
    return before, after, fn, {}


def _gen_running_count_to_incremental(rng: random.Random) -> TemplateResult:
    """O(n^2) -> O(n) -- recounting how many elements-so-far satisfy a
    condition by rescanning the growing window every index, instead of
    incrementing a running counter. See _gen_running_sum_to_cumulative's
    docstring for why this family exists."""
    fn = rng.choice(["running_count_above", "count_so_far", "tally_matching"])
    nums = rng.choice(["nums", "values", "readings"])
    threshold = rng.choice(["threshold", "limit", "cutoff"])
    acc = rng.choice(["result", "output", "counts"])
    before = (
        f"def {fn}({nums}, {threshold}):\n"
        f"    {acc} = []\n"
        f"    for i in range(len({nums})):\n"
        f"        window = {nums}[:i + 1]\n"
        f"        count = 0\n"
        f"        for value in window:\n"
        f"            if value > {threshold}:\n"
        f"                count = count + 1\n"
        f"        {acc}.append(count)\n"
        f"    return {acc}\n"
    )
    after = (
        f"def {fn}({nums}, {threshold}):\n"
        f"    {acc} = []\n"
        f"    count = 0\n"
        f"    for value in {nums}:\n"
        f"        if value > {threshold}:\n"
        f"            count = count + 1\n"
        f"        {acc}.append(count)\n"
        f"    return {acc}\n"
    )
    return before, after, fn, {}


def _gen_pop0_to_deque(rng: random.Random) -> TemplateResult:
    """O(n^2) -> O(n) -- list.pop(0) shifts every remaining element (O(n)
    per call), so draining a list this way is O(n^2). deque.popleft() is
    O(1). Output order (FIFO) is identical either way."""
    fn = rng.choice(["drain_queue", "process_fifo", "consume_all"])
    items = rng.choice(["items", "queue", "tasks"])
    acc = rng.choice(["result", "output", "processed"])
    before = (
        f"def {fn}({items}):\n"
        f"    {items} = list({items})\n"
        f"    {acc} = []\n"
        f"    while {items}:\n"
        f"        {acc}.append({items}.pop(0))\n"
        f"    return {acc}\n"
    )
    after = (
        "from collections import deque\n\n"
        f"def {fn}({items}):\n"
        f"    dq = deque({items})\n"
        f"    {acc} = []\n"
        f"    while dq:\n"
        f"        {acc}.append(dq.popleft())\n"
        f"    return {acc}\n"
    )
    return before, after, fn, {}


FAMILIES: List[TransformationFamily] = [
    TransformationFamily("membership_to_set", "O(n^2) -> O(n)", _gen_membership_to_set),
    TransformationFamily("index_to_dict", "O(n^2) -> O(n)", _gen_index_to_dict),
    TransformationFamily("string_concat_to_join", "O(n^2) -> O(n)", _gen_string_concat_to_join),
    TransformationFamily("naive_fib_to_memo", "O(2^n) -> O(n)", _gen_naive_fib_to_memo),
    TransformationFamily("pair_sum_count_to_hashmap", "O(n^2) -> O(n)", _gen_pair_sum_count_to_hashmap),
    TransformationFamily("sort_in_loop_to_once", "O(k*n log n) -> O(n log n + k)", _gen_sort_in_loop_to_once),
    TransformationFamily("minmax_in_loop_to_running", "O(n^2) -> O(n)", _gen_minmax_in_loop_to_running),
    TransformationFamily("pop0_to_deque", "O(n^2) -> O(n)", _gen_pop0_to_deque),
    # Added to address the diagnosed weak spot (see
    # _gen_running_sum_to_cumulative's docstring) -- both TRAINED, not
    # held out, so they teach the "running accumulator" concept that no
    # earlier family covered explicitly.
    TransformationFamily("running_sum_to_cumulative", "O(n^2) -> O(n)", _gen_running_sum_to_cumulative),
    TransformationFamily("running_count_to_incremental", "O(n^2) -> O(n)", _gen_running_count_to_incremental),
]


def generate_pairs(instances_per_family: int = 25, seed: int = 42) -> List[TrainingPair]:
    """Every instance is a fresh call to its family's template with its own
    rng state, so instances differ lexically (names) but not structurally."""
    rng = random.Random(seed)
    pairs: List[TrainingPair] = []
    for family in FAMILIES:
        for _ in range(instances_per_family):
            before, after, func_name, custom_values = family.generate(rng)
            pairs.append(TrainingPair(
                family=family.name,
                complexity_change=family.complexity_change,
                before=before,
                after=after,
                function_name=func_name,
                custom_values=custom_values,
            ))
    return pairs
