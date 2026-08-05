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
Confidence             96%

Why
  • two nested loops — each element is compared against every other
  • a linear scan inside a loop (`in` against a list) — this is a hidden quadratic

How to make it faster
  Convert the list being searched into a set before the loop. Membership
  testing drops from O(n) to O(1), taking the whole function from
  quadratic to linear.
```

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

| Metric | Result |
|---|---|
| Grouped 5-fold cross-validation | **98.1%** |
| Held-out benchmark (22 DSA functions) | **22/22** |
| Stratified 5-fold cross-validation | 100.0% |
| Training corpus | 530 samples from 106 base algorithms |

**Read the grouped number, not the stratified one.** The corpus expands each
base algorithm into structural variants — renamed identifiers, extra
statements, extra branches — so that the model learns to ignore things that
do not affect complexity. Under a plain stratified split, variants of the
same algorithm land in both the train and test folds, and the model scores
100% by recognising near-copies. Grouped cross-validation forces every
variant of an algorithm into the same fold, so the test set contains only
algorithms the model has never seen. That is the number worth quoting.

The held-out benchmark in `app/ml/benchmark.py` is a third check: 22
functions written independently and deliberately kept out of the corpus.

### Per-class breakdown (grouped CV)

```
                  0     1     2     3     4     5     6
       O(1)      40     .     .     .     .     .     .     100%
   O(log n)       .    55     5     .     .     .     .      92%
       O(n)       .     .   170     .     .     .     .     100%
 O(n log n)       .     .     .    64     .     .     1      98%
      O(n²)       .     .     5     5    85     .     .      89%
     O(n³+)       .     .     .     .     .    45     .     100%
     O(2^n)       .     .     .     .     .     .    55     100%
             (rows = actual, columns = predicted)
```

Reproduce with `python -m app.ml.evaluate`.

### Which features carry the signal

```
max_loop_depth              0.170
has_sorting_call            0.139
nested_loop_count           0.119
loop_count                  0.117
while_loop_halves           0.117
max_self_calls_per_path     0.114
has_recursion               0.057
...
share taken by known-irrelevant features: 7.9%
```

That last line is the check that the corpus design worked. `branch_count`,
`has_subscript` and `total_statements` are deliberately kept in the feature
vector even though they carry no asymptotic signal — the corpus varies them
while holding the label fixed, so a well-trained forest should learn to
ignore them. It does.

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
    "before": "O(n²)", "after": "O(n²)", "confidence": 0.96,
    "explanation": ["two nested loops — ..."],
    "suggestion": "Convert the list being searched into a set ..."
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

**Behaviour is tested by execution, not inspection.**
`test_refactor_preserves_behaviour` runs the original and the refactored code
with real inputs and compares results. Structural assertions cannot catch a
refactoring that produces plausible-looking but broken code; running it can.
That test found three real bugs on its first run.

---

## Known limitations

Stated plainly, because a tool that overstates its confidence is worse than
one that admits its edges:

- Complexity is reported **per module**, not per function. A file with a
  linear helper and a quadratic main function reports the quadratic.
- **No interprocedural analysis.** If `main()` calls a helper that sorts, the
  cost of that sort is not propagated into `main`'s complexity.
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
