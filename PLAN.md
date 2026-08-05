# ACRR — 7-Day Completion Plan

**Written:** 5 August 2026
**Target ship date:** 12 August 2026

---

## Where the project actually stands

The complexity engine — the feature the whole app exists for — was the weakest
part of the codebase, and it is now the strongest. That work is done. What
remains is mostly *verification of things that have never been run*, plus
deployment.

### What changed in this session

| File | Change |
|---|---|
| `app/ml/features.py` | **New.** 18-feature structural extractor, split out so training and prediction use one code path |
| `app/ml/corpus.py` | **New.** 370 labeled samples from 74 real algorithms, stored as Python source |
| `app/ml/benchmark.py` | **New.** 22 held-out DSA functions, deliberately not in the corpus |
| `app/ml/evaluate.py` | **New.** Grouped cross-validation, confusion matrix, feature importances |
| `app/ml/complexity_predictor.py` | **Rewritten.** Same RandomForest; trains from the corpus; adds explanations |
| `app/models.py` | `ComplexityPrediction` gains `explanation` and `suggestion` |
| `app/main.py` | HTTP error handler fixed |
| `app/refactor/orchestrator.py` | `loop_opts` enabled at the default level |
| `requirements.txt` | Rewritten — was missing pydantic and all ML deps |
| `frontend/.../RefactorPanel.jsx` + `.module.css` | Complexity is now a first-class tab with reasoning |
| `app/middleware.py` | Dead file — **delete manually**, this session could not delete files on your disk |

### Measured results

| Metric | Before | After |
|---|---|---|
| Held-out benchmark (22 real DSA functions) | 18/22 (82%) | **22/22 (100%)** |
| Grouped cross-validation (unseen algorithms) | not measured | **96.8%** |
| Typical confidence on a correct answer | 0.49 – 0.66 | **0.83 – 1.00** |
| Training samples | 44 hand-typed vectors | **370 real functions** |
| Contradictory training samples | present (silently broke O(n²)) | **0** |

The four cases that previously failed — `merge_sort`, `quicksort`, `bfs_graph`,
`list_in_loop` — all pass now, because comprehensions are counted as loops,
divide-and-conquer recursion is distinguished from exponential recursion, and
`x in y` is resolved against whether `y` is a set or a list.

---

## The one decision I made for you

**The Gemini AI layer stays, but off the critical path.**

Reasoning:

*For keeping it:* it already degrades gracefully — `ai_available: false` and an
empty array, no crash. The blocker was always quota, never code. It genuinely
adds something the rule engine cannot: prose explanation of semantic intent.
And removing it would mean deleting working code and a stated feature.

*For not prioritising it:* it is the only part of the system that depends on a
third party, a network call, a rotating key format, and a daily quota. On a
7-day deadline that is exactly the wrong thing to have on the critical path. A
demo that fails because Google's quota reset at midnight Pacific is a bad day.

*Resolution:* leave the code as-is, verify the fallback path deliberately (Day
4), and make the UI honest about when AI is off. Do **not** let any demo, test,
or deployment step depend on a live Gemini response. If quota happens to be
available on demo day, it is a bonus, not a requirement.

---

## Day 1 — Rebuild the environment and confirm the new engine

The most likely failure mode this week is that something works here and not on
your machine. Close that gap first.

```
cd backend
rmdir /s /q .venv
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then:

- [ ] `python -m app.ml.evaluate` — expect grouped CV ≈ 96.8%, benchmark 22/22
- [ ] `python -m pytest tests -q` — the existing 24 tests must still pass
- [ ] `python -m app.main` and POST a nested-loop function to `/refactor`;
      confirm `complexity.explanation` comes back populated
- [ ] Delete `backend/app/middleware.py`
- [ ] Delete the old `complexity_rf.pkl` if it is still there — the new loader
      detects the stale fingerprint and retrains automatically (~1.5s), but
      deleting it makes that explicit
- [ ] Confirm `backend/.env` is **not** in git: `git log --all -- backend/.env`.
      If it ever was committed, rotate the Gemini key — it is in history forever.

**Done when:** a fresh clone installs and evaluates cleanly with one command.

---

## Day 2 — Test the refactor rules, especially `loop_opts`

This is the highest-risk item in the whole project and it needs a full day.

`loop_opts` was configured to run only at `level=high`, but the frontend sends
`level=medium`. It therefore **never ran in normal use** — which means it has
never been meaningfully tested. This session enabled it at the default level, so
it is now live on every request. That is correct, but it is new exposure.

This is also the exact category of code that was stubbed out (`return node`,
doing nothing) in v1. Treat every rule as guilty until proven otherwise.

- [ ] Write one targeted test per transform in `loop_opts.py` — a case that
      should fire, and a lookalike case that should **not**
- [ ] Property test: for 50 corpus samples, assert the refactored output
      (a) parses, and (b) is not semantically absurd
- [ ] Verify `range(len(x))` → `enumerate(x)` only fires when the index is used
      solely for indexing that one sequence
- [ ] Same treatment for `simplify_conditions`, which is currently only
      "reported working"
- [ ] Add a regression test asserting clean code is returned byte-identical

**Done when:** every refactor rule has a fires/does-not-fire test pair, and you
have deliberately tried to make each one produce wrong code.

---

## Day 3 — Review engine: prove the three linters actually work

`flake8`, `radon` and `bandit` are wired in, and the endpoint returns 200. That
is not the same as them producing correct output — and all three fail *silently*
by design (`except ImportError: return []`), which is right for robustness and
dangerous for confidence.

- [ ] Write a code sample with a **known** flake8 violation; assert the exact
      rule code appears
- [ ] Known-vulnerable sample (`eval()`, `subprocess(shell=True)`, hardcoded
      password); assert bandit reports it with the right severity
- [ ] A function with cyclomatic complexity > 10; assert `CC001` fires
- [ ] Add a startup check to `/health` reporting which analysers are actually
      importable, so a missing linter is visible instead of silent
- [ ] Re-check grading: messy code should land C or below, clean code A

**Done when:** each analyser has a test that fails if the analyser silently
disappears.

---

## Day 4 — Frontend, end to end

This is the least-verified part of the project. "Builds clean" has been the only
claim made about it, and building is not running.

- [ ] `npm install && npm run dev`, then actually use it
- [ ] Paste the example code, click Review, click Refactor — check both panels
      render real data
- [ ] Verify the new **Complexity** tab: Big-O, confidence bar, "Why" bullets,
      and the suggestion all display
- [ ] Test with AI off — confirm the "AI unavailable" state renders and nothing
      breaks (this is the graceful-degradation check; do it deliberately)
- [ ] Error states: empty input, syntax error, 100k+ character paste
- [ ] Confirm the Vite proxy points at port 8000 and matches `main.py`
- [ ] Check the Monaco editor works on a slow load and does not blank the page

**Done when:** you have used the app as a user for 20 minutes without opening
the console.

---

## Day 5 — Corpus expansion and honest numbers

96.8% grouped CV on 74 base algorithms is good. It is also a small corpus, and
you should know where it breaks before someone else finds out.

- [ ] Add 15–25 more base algorithms to `corpus.py`, biased toward what your
      users will actually paste: string manipulation, matrix problems, linked
      list traversal, sliding window, two pointers, heap operations
- [ ] Add `O(log n)` cases — it is the weakest class in the confusion matrix
      (83%, occasionally confused with `O(2^n)`)
- [ ] Re-run `evaluate.py`. **If grouped CV drops, that is good information**,
      not a failure — it means the new samples exposed a real gap
- [ ] Write down the known limitations honestly:
      - complexity is reported per-module, not per-function
      - amortised costs (dynamic array growth) are not modelled
      - input-dependent complexity (quicksort worst case) reports the average
      - no interprocedural analysis — a helper function's cost is not
        propagated into its caller

**Done when:** the numbers in your report are the grouped-CV ones, not the
flattering stratified ones.

---

## Day 6 — Deployment

- [ ] Backend to Render free tier. Add a `gunicorn` entry to requirements and a
      `Procfile`; Flask's dev server is not for deployment
- [ ] **Watch the cold start.** Render free tier sleeps. First request after
      sleep pays model training (~1.5s) on top of container wake. Either commit
      the trained `.pkl` or warm the model at import time rather than on first
      request
- [ ] Check the memory ceiling — scikit-learn plus a 300-tree forest is not
      free on a 512MB instance. If it is tight, drop `n_estimators` to 150 and
      re-run `evaluate.py` to confirm accuracy holds
- [ ] Frontend to Vercel; set `VITE_API_BASE_URL` to the Render URL
- [ ] Restrict CORS to your frontend origin — `CORS(app)` currently allows
      everything
- [ ] Set `GEMINI_API_KEY` as an environment variable in Render, never in the repo
- [ ] Smoke-test the deployed app from a different machine

**Done when:** a stranger can use the URL without you present.

---

## Day 7 — Report, demo, buffer

- [ ] README rewrite: what it does, how to run it, measured accuracy with the
      methodology stated
- [ ] Include the confusion matrix and feature importance table from
      `evaluate.py` — they are genuinely good report material
- [ ] Write up the ML story properly: the model is a RandomForest over 18
      structural AST features, trained on 370 samples derived from 74
      algorithms, evaluated with grouped 5-fold cross-validation so that no
      variant of a training algorithm appears in its own test fold
- [ ] Prepare three demo inputs: a nested-loop quadratic, a naive recursive
      Fibonacci, and a divide-and-conquer sort. Each shows a different part
      of the engine
- [ ] **Keep this day as buffer.** Something on Days 2–4 will overrun.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `loop_opts` produces incorrect refactorings now that it runs by default | **High** — it has never been exercised | Day 2 is dedicated to this. If tests find real bugs, revert it to HIGH-only and ship without it |
| Frontend has a blocking bug nobody has seen | Medium — it has never been run | Day 4, early enough to fix |
| Render free tier memory limit | Medium | Reduce `n_estimators`; re-verify accuracy |
| Gemini quota blocks the demo | Low impact by design | Never demo the AI path; fallback is already graceful |
| Corpus expansion drops accuracy | Medium | This is information, not failure. Report the honest number |

---

## What I would cut if you fall behind

In order, the first things to drop:

1. **The AI suggestion layer.** Set `use_ai` default to `false`. The app is
   complete without it.
2. **Corpus expansion (Day 5).** 96.8% on 74 algorithms is a defensible result.
3. **`loop_opts`.** Revert to HIGH-only. Four working refactor rules that are
   tested beat five where one is unverified.

What must **not** be cut: Day 1 (environment), Day 2 (refactor correctness —
an app that silently breaks user code is worse than one that does nothing), and
Day 4 (the frontend, because an untested UI is an untested product).
