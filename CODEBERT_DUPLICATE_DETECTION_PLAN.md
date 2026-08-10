# CodeBERT near-duplicate detection — 3-day implementation plan

**Status:** Not started. Written 10 August 2026, for later execution — deployment
and the AI provider issue take priority first (see "Priority note" at the end).

---

## Why CodeBERT, and why only this use case

Considered CodeBERT and CodeT5 for expanding ACRR's ML beyond the existing
RandomForest complexity predictor. Conclusion: **do not** use either for
smell/bug classification or generation — that duplicates the existing Gemini
AI layer with a strictly weaker, older model (CodeBERT: 125M params, 2020;
CodeT5-base: 220M, 2021, vs. a modern general-purpose LLM already wired in).
Building a properly-validated classifier to this project's own bar (the bar
the RandomForest complexity model already proved necessary: 149 individually
verified algorithms, contradiction checks, nested CV, an adversarial audit
tool) is a multi-week effort for a capability that already exists.

**The one legitimate exception**: semantic near-duplicate detection.
`SMELL006` (in `backend/app/review/smells.py`) currently only catches
*exact* textual duplicate blocks. It misses semantically-equivalent code that
differs by variable renaming or statement reordering. CodeBERT's pretrained
embeddings (no fine-tuning needed) are the standard tool for this in the
literature (BigCloneBench, POJ-104 clone detection) — inference-only, no
dataset to collect or validate.

**Why CodeBERT over CodeT5 for this specific task**: CodeBERT is
encoder-only, so you get a fixed-size embedding directly with no unused
decoder weights to load. Its MLM + replaced-token-detection pretraining on
bimodal NL-PL pairs is built for producing a similarity-meaningful embedding
space. CodeT5's identifier-aware denoising objective targets
reconstruction/generation, not similarity. CodeT5 would be the right choice
if the task were generation (fixes, explanations) — but that task is the one
already ruled out as duplicating Gemini.

**Scope limit, stated plainly**: this only compares functions *within a
single submitted snippet* against each other. ACRR analyzes one paste at a
time with no persistent corpus, so there's no cross-session or
whole-codebase duplicate detection here. Value is real but narrow: useful
when someone pastes multiple functions and two are near-identical, useless
on a single-function submission (the common case).

---

## Day 1 — Core detection module (no API yet)

**Goal**: `find_near_duplicates()` works correctly in isolation.

1. Add `torch` (CPU-only wheel) and `transformers` to `requirements.txt`, in
   a clearly separate, commented section (why they're heavy — mirrors how
   `google-genai` is documented as optional-at-runtime).
2. New file `backend/app/ml/duplicate_detector.py`:
   - Lazy-load `microsoft/codebert-base` tokenizer + model as a
     module-level singleton on first call — same `_get_model()` caching
     pattern already used in `complexity_predictor.py`. Must NOT load at
     process startup (`--preload` in gunicorn already loads the RandomForest
     there; adding ~500MB more at startup on a memory-constrained instance
     is the wrong default).
   - `embed_function(code: str) -> np.ndarray` — tokenize, forward pass,
     mean-pool the last hidden state over the attention mask (not a raw
     mean — padding tokens would dilute the vector).
   - `find_near_duplicates(functions, threshold=0.92) -> List[DuplicatePair]`
     — pairwise cosine similarity across all functions in one submission.
     Skip functions under ~3 statements (trivially "similar" one-line
     getters/setters would otherwise flood results with noise).
3. Calibrate the threshold by hand against a handful of deliberately
   constructed cases: a true near-duplicate (renamed variables, reordered
   independent statements) vs. two genuinely different but shaped-alike
   functions (e.g. two different one-line property getters). **This is
   spot-checking, not formal validation** — say so explicitly in the code
   comment. No nested CV, no audit tool, unlike the complexity model.
4. Unit tests mirroring `tests/test_ai_client.py`'s shape: pure-function
   tests for the similarity math, integration test skipped (not failed) if
   `torch` isn't installed — same "skip rather than fail when tool not
   installed" convention `test_review_engine.py` already uses for
   `flake8`/`bandit`.

## Day 2 — API + graceful degradation

**Goal**: a working endpoint that behaves correctly whether or not the
model is available.

1. New Pydantic models in `backend/app/models.py`: `DuplicatePair` (two
   function names, line numbers, similarity score), `DuplicateCheckResponse`.
2. New endpoint `POST /duplicates` in `backend/app/main.py` — kept separate
   from `/review` and `/refactor`, not bolted onto either, since it's an
   unrelated, optional, heavier operation.
3. Graceful degradation, matching the AI layer's exact pattern: catch
   `ImportError` / timeout / model-load failure at call time, return
   `available: false` rather than a 500. Never let a missing dependency take
   the endpoint down.
4. Wire into `/health` so it's visible which analysers are actually usable
   — consistent with the existing convention there.
5. Backend tests: syntax gate, graceful-degrade path, and a real detection
   test on a crafted duplicate pair (two functions, same logic, renamed
   variables) — the test that actually proves the feature works, not just
   that it doesn't crash.

## Day 3 — Frontend + end-to-end verification + docs

**Goal**: a user can click a button and see real results, full suite green.

1. Frontend: a third button — **"Check for duplicates,"** deliberately not
   labeled "Use ML" or similar. The RandomForest complexity model already
   runs ML on every request by default; a button literally called "ML" next
   to that would confuse users about what's actually being toggled.
2. Loading state that says up front "first check may take longer while the
   model loads" — cold-start on a memory/CPU-constrained free-tier host
   (model download + load) is realistically 10-30+ seconds, and silently
   hanging there reads as broken.
3. Full verification: `pytest`, a `verify_changes.py`-style end-to-end
   addition, one manual test against real multi-function code with an
   actual near-duplicate in it.
4. Update `README.md` / `HANDOFF.md`: document the feature as
   **experimental**, state the threshold is hand-calibrated not validated,
   and record the deployment decision explicitly — recommend keeping it
   env-gated (installed/enabled only where chosen) rather than forced onto
   a memory-constrained free-tier instance, given HANDOFF.md already flags
   memory as tight for the *existing*, much smaller RandomForest.

---

## Risks that could push this past 3 days

- **Environment friction.** The same week this plan was written hit two
  unrelated SSL/cert surprises on this machine (`git push`, Gemini
  connectivity). A ~2GB `torch` install is exactly the kind of dependency
  that surfaces a new environment quirk. Budget slack in Day 1, don't
  assume a clean install.
- **Threshold tuning taking longer than expected** if the first few
  hand-picked test cases don't cleanly separate true/false positives — the
  one genuinely open-ended part of the plan.

---

## Priority note (why this hasn't been started)

Written the same day as: (a) fixing an SSL certificate issue blocking the
existing Gemini AI layer, and (b) discovering the configured Gemini API key
has zero quota allocated on its project. Deployment (Render + Vercel) is
also still pending, with an imminent ship target. This plan is additional
scope competing with both of those. Recommended sequencing: resolve the AI
provider situation and ship the app first; build this once there's a
concrete case of a missed duplicate in real usage, not speculatively.
