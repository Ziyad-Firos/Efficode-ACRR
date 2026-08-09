# ACRR — Automated Code Review & Refactoring

Paste Python code. Get back its time complexity, a review, and an optimised
version with an explanation of every change.

Built for DSA code specifically — the kind of thing you write for interview
practice or a data structures course, where "is this O(n) or O(n²)?" is the
question that actually matters.

---

## What it does

**Predicts time complexity.** A RandomForest classifier over 18 structural
features extracted from the AST. Returns a Big-O class, a confidence score,
the structural evidence behind the prediction, and the single change that
would most improve it.

```
Time complexity        O(n²)
Confidence             99%

Why
  • 'find_duplicates' is the slowest of 2 functions and sets the overall complexity.
  • two nested loops — each element is compared against every other
  • a linear scan inside a loop (`in` against a list) — this is a hidden quadratic

Per function
  find_duplicates    line 1    O(n²)   99%   slowest
  calculate_stats    line 10   O(n)    92%

How to make it faster
  Convert the list being searched into a set before the loop. Membership
  testing drops from O(n) to O(1), taking the whole function from
  quadratic to linear.
```

Each function is predicted on its own and the slowest one sets the headline.
That matters more than it sounds: analysing the file above as a single unit
blends a quadratic function with a linear one and returns a hedged answer at
52% confidence. Split apart, the same two functions come back at 99% and 92%.

**Reviews.** flake8 (style), radon (cyclomatic complexity), bandit
(security), plus custom AST checks for code smells. Produces a weighted A–F
grade.

**Refactors.** Deterministic AST transforms — constant folding, dead code
removal, unused variable removal, condition simplification, loop
optimisation — with a unified diff and a plain-English description of each
change. An optional Gemini layer adds semantic suggestions; the app degrades
to rules-only when it is unavailable.

---

## Measured accuracy

Read this section before quoting any number from it. Several obvious-looking
metrics for this model are misleading, and they are labelled as such.

| Metric | Result | Trust it? |
|---|---|---|
| **Nested grouped cross-validation** | **97.6%** | **Yes — this is the number to quote** |
| Accuracy on genuinely unseen patterns | 8/8 | Yes, but the sample is small |
| Grouped 5-fold cross-validation | 97.3% | Mostly — tuned and evaluated on the same folds |
| Held-out benchmark | 23/23 | **No — 83% contaminated, see below** |
| Stratified 5-fold cross-validation | 100.0% | **No — variants leak across folds** |

Training data: **149 independent algorithms**, expanded to 745 samples by
augmentation. Quote 149. The extra samples are structural no-ops — renamed
identifiers, added statements, added branches — that teach the model which
features to ignore. They are not new evidence about complexity.

Run `python -m app.ml.audit` to reproduce every claim below. Use the exact
versions in [`backend/requirements-lock.txt`](backend/requirements-lock.txt)
when doing so — `>=` ranges in `requirements.txt` are for humans installing
the app, not for reproducing these numbers. This already caused a real
discrepancy: scikit-learn 1.8 produced 98.1% grouped CV and 1.9 produced
97.4% on identical code.

### Why the flattering numbers are wrong

**The stratified split leaks.** Variants of one algorithm land in both the
train and test folds, so the model scores 100% by recognising near-copies.
Grouped cross-validation forces every variant of an algorithm into the same
fold, so the test set holds only algorithms the model has never seen.

**The benchmark is 83% contaminated.** Of 23 benchmark cases, 19 have a
feature vector *identical* to a training algorithm — `bubble_sort` in the
benchmark produces the same 23 numbers as `bubble_sort` in the corpus. Those
cases cannot fail unless the model is broken. The benchmark is a smoke test,
not a generalisation estimate, and is kept for that purpose.

**Tuning and evaluating on the same folds inflates the score.** `n_estimators`
was chosen by maximising grouped CV, which was then reported as a result.
Nested cross-validation re-selects hyperparameters inside each fold, so the
test data never influences the choice. That is the 97.6% figure — with the
expanded corpus below, tuning optimism is now -0.3 pts (grouped CV slightly
*under*-reports nested CV, itself a sign of a small-sample fold, not bias).

### Per-class breakdown (grouped CV)

```
                  0     1     2     3     4     5     6
       O(1)      60     .     .     .     .     .     .     100%
   O(log n)       .    55     2     .     .     .     3      92%
       O(n)       .     .   230     .     .     .     .     100%
 O(n log n)       .     .     .    98     7     .     .      93%
      O(n²)       .     .     8     .   127     .     .      94%
     O(n³+)       .     .     .     .     .    90     .     100%
     O(2^n)       .     .     .     .     .     .    65     100%
             (rows = actual, columns = predicted)
```

`O(n log n)` and `O(n³+)` were the weakest and thinnest classes (13 and 9
independent algorithms). Both were expanded with real algorithms — heap-based
selection, k-way merges, interval scheduling, Gaussian elimination, brute-force
subarray/substring search, Bellman-Ford relaxation, and more — bringing them
to 21 and 18 independent algorithms respectively.

### Calibration

When the model reports 95% confidence, it is right 100% of the time; at 50–70%
confidence it is right 94% of the time. Mean confidence 86% against 97% actual
accuracy — the model is **under-confident**, which is the safe direction. A
low confidence figure is a real signal that the code is unusual, not noise.

### What the audit checks

`app/ml/audit.py` exists to attack the numbers rather than produce them:
train/test leakage, effective sample size, hyperparameter selection bias,
probability calibration, robustness on deliberately unfamiliar code, and a
list of corpus labels that are genuinely arguable. It was written because
every one of those is a standard way for an ML result to be quietly wrong.

It has already earned its place: after string-concatenation detection was
added, the model started disagreeing with a benchmark case labelled `O(n)`.
The model was right. `out = ch + out` in a loop copies the whole accumulated
string every iteration and is quadratic — the hand-written ground-truth label
was wrong. It is now labelled `O(n²)`, with a linear `str.join` version added
alongside for contrast.

### Which features carry the signal

```
max_loop_depth              0.144
has_sorting_call            0.128
while_loop_halves           0.117
loop_count                  0.104
nested_loop_count           0.098
max_self_calls_per_path     0.077
has_recursion               0.075
...
share taken by known-irrelevant features: 8.1%
```

One feature, `has_convergence_flag_while`, deliberately carries **zero**
importance (0.000). It detects the "while flag: flag = False; ...; flag =
True" idiom — early-exit bubble sort, Bellman-Ford relaxation, any
fixed-point pass — and exists only so the explanation layer can flag its own
uncertainty. The same shape is genuinely O(n²) for one real algorithm and
O(n³) for another (verified empirically on both — see `check_labels()` in
`audit.py`), so the forest correctly learned not to lean on it for the label.
Confidence should come from `explain()`'s caveat text here, not the class
probability.

That last line checks the corpus design. `branch_count`, `has_subscript` and
`total_statements` are kept in the vector despite carrying no asymptotic
signal — the corpus varies them while holding the label fixed, so a
well-trained forest should learn to ignore them. It does.

---

## Quick start

Requires **Python 3.12**. Not 3.13+, and definitely not a 3.15 beta —
scikit-learn and numpy have no prebuilt wheels there.

```bash
# backend
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python -m app.main              # http://localhost:8000
```

```bash
# frontend, in a second terminal
cd frontend
npm install
npm run dev                     # http://localhost:3000
```

Or the whole stack at once:

```bash
docker compose up --build
```

The AI layer is optional. Without a `GEMINI_API_KEY` the app runs fully on
its deterministic engines and reports `ai_available: false`.

---

## Verifying it

```bash
cd backend
python -m pytest tests -q          # unit + behaviour tests
python -m app.ml.evaluate          # complexity model accuracy
python -m app.ml.audit             # attack those numbers: leakage, bias, calibration
python verify_changes.py           # end-to-end API checks
```

Each test file also runs standalone without pytest:

```bash
python tests/test_refactor_rules.py
python tests/test_review_engine.py
```

`/health` reports which analysers are actually usable, so a missing tool is
visible rather than silently producing an empty review:

```json
{
  "status": "ok",
  "analysers": {
    "style_flake8": true,
    "security_bandit": true,
    "complexity_radon": true,
    "smells_ast": true,
    "ml_sklearn": true
  },
  "degraded": []
}
```

---

## API

### `POST /review`

```json
{ "code": "def f(x):\n    return x + 1" }
```

Returns issues from all four analysers plus an A–F quality score.

### `POST /refactor`

```json
{ "code": "...", "level": "medium", "use_ai": true }
```

```json
{
  "refactored_code": "...",
  "diff": "...",
  "applied_rules": [
    {"rule": "dead_code_after_terminal", "line": 4,
     "description": "...", "applied": true}
  ],
  "complexity": {
    "before": "O(n²)", "after": "O(n²)", "confidence": 0.99,
    "explanation": ["two nested loops — ..."],
    "suggestion": "Convert the list being searched into a set ...",
    "functions": [
      {"name": "find_duplicates", "line": 1, "complexity": "O(n²)",
       "confidence": 0.99, "explanation": ["..."], "is_dominant": true},
      {"name": "calculate_stats", "line": 10, "complexity": "O(n)",
       "confidence": 0.92, "explanation": ["..."], "is_dominant": false}
    ]
  },
  "ai_suggestions": [],
  "ai_available": false
}
```

`applied: false` marks advice that was reported but deliberately **not**
auto-applied, because applying it could change behaviour. See below.

### `GET /health`

Liveness, AI configuration, and analyser availability.

---

## Design decisions worth knowing

**Nothing submitted is ever executed.** Every engine works statically over
`ast.parse()`. No `eval`, no `exec`, no subprocess running user code. This
removes an entire class of risk at the architecture level rather than
guarding against it case by case.

**Training data is source code, not feature vectors.** An earlier version
trained on 44 hand-typed rows of numbers. Two problems followed: the rows
could drift out of sync with the extractor, and structurally identical
algorithms produced identical vectors — a row for BFS labelled O(n) was
byte-identical to one for bubble sort labelled O(n²), which silently
destroyed the O(n²) class. `corpus.py` now stores real Python, and features
are extracted by the live extractor at training time.

**Some refactors are reported, not applied.** `if x == True:` looks like it
should become `if x:` — but `2 == True` is `False` while `if 2` is truthy, so
that rewrite silently changes behaviour for every non-boolean value. It is
applied only when the left side is provably a bool, and reported as advice
otherwise. The same reasoning guards the loop-to-comprehension transform,
which is refused when the accumulator is read inside the loop, when the loop
variable is used afterwards, or when the target shadows the accumulator —
each of those produces code that raises `NameError` at runtime.

**Complexity is predicted per function, not per file.** A file is not a
meaningful unit of asymptotic analysis — its loop structure is the union of
every function in it. Predicting each function separately and reporting the
slowest is both more accurate and more useful, since it names which function
to fix.

**Behaviour is tested by execution, not inspection.**
`test_refactor_preserves_behaviour` runs the original and the refactored code
with real inputs and compares results. Structural assertions cannot catch a
refactoring that produces plausible-looking but broken code; running it can.
That test found three real bugs on its first run.

---

## Known limitations

Stated plainly, because a tool that overstates its confidence is worse than
one that admits its edges:

- **The training corpus is small.** 149 algorithms, hand-labelled by one
  person. `O(n³+)` has 18 examples and `O(n log n)` has 21 — thin classes,
  though better than before. Expect worse behaviour on classes that thin.
- **Convergence loops are structurally undecidable from syntax.** A
  `while flag: flag = False; for i: for j: ... flag = True` loop can be
  O(n²) (an early-exit comparison sort, converging in 1-2 passes) or O(n³)
  (Bellman-Ford relaxation over an adjacency matrix, needing up to n passes)
  — same shape, opposite true cost, verified empirically on both. No
  structural feature can resolve this; it would require reasoning about what
  the loop body computes. The model does not attempt to pick a label for
  this shape either way — `explain()` surfaces the ambiguity instead.
- **Labels encode one opinion.** Quicksort is labelled `O(n log n)`, not its
  `O(n²)` worst case. BFS is labelled `O(n)` though it is really `O(V+E)`.
  `n` is not even consistently defined — for `gcd` it is the magnitude of a
  number, for sorting it is an element count. The model cannot tell these
  apart.
- **No execution, ever.** Complexity is inferred from structure alone. Nothing
  is measured empirically, so a data-dependent cost the structure does not
  reveal will be missed.

- **No interprocedural analysis.** If `main()` calls a helper that sorts, the
  cost of that sort is not propagated into `main`'s complexity. Each function
  is judged on the structure written inside it.
- **Amortised costs are not modelled.** Repeated `list.append` is treated as
  O(1) per call, which is the amortised truth but not the worst case.
- **Input-dependent complexity reports the typical case.** Quicksort is
  reported O(n log n), not its O(n²) worst case.
- **Space complexity is not predicted at all.** Time only.
- The corpus is 106 algorithms. It covers common DSA patterns well and will
  be less reliable on unusual control flow — generators driving state
  machines, heavy metaprogramming, recursion through mutual calls rather
  than self-calls.

---

## Project layout

```
backend/
  app/
    main.py                  Flask app — the only entry point
    models.py                Pydantic schemas; the API contract
    parser.py                safe ast.parse wrapper
    review/
      style.py               flake8
      complexity.py          radon (cyclomatic — not Big-O)
      security.py            bandit
      smells.py              custom AST checks
    refactor/
      orchestrator.py        runs rules in order, rolls back on failure
      rules/                 one module per transform
      ai_client.py           Gemini, with validation and fallback
    ml/
      corpus.py              530 labeled samples from 106 algorithms
      features.py            18-feature AST extractor
      complexity_predictor.py  RandomForest + explanations
      benchmark.py           22 held-out functions
      evaluate.py            grouped CV, confusion matrix, importances
      audit.py               leakage, selection bias, calibration, robustness
  tests/
frontend/
  src/components/            CodeEditor, ReviewPanel, RefactorPanel, ScoreCard
```

**Cyclomatic complexity (radon) and Big-O complexity (the ML model) are
different things** and both are reported. Cyclomatic complexity counts
independent paths through a function — a readability and testability signal.
Big-O describes how runtime grows with input size. A function can be
cyclomatically simple and asymptotically terrible; `for i in nums: for j in
nums: pass` is the whole point.

---

## Deployment

`render.yaml` deploys the backend to Render's free tier;
`frontend/.env.example` shows the one variable the frontend needs.

Two things to get right:

- **Set `ALLOWED_ORIGINS`** to the deployed frontend URL. Unset, the backend
  accepts requests from any origin and logs a warning at startup.
- **Keep `--preload` in the gunicorn command.** It trains the complexity
  model once in the parent process before workers fork. Without it, every
  worker retrains on its first request — and on a free tier that sleeps,
  that request is already paying for the container to wake up.
