# ACRR — Session report: architecture and work log

**Written:** 10 August 2026, covering the full working session from initial
handoff through the current state below.
**Repo:** `Efficode-ACRR`, branch `dev`, HEAD at `d89ce17`.
**Companion doc:** `HANDOFF.md` is the living day-to-day status doc (steps
1–7, ordered next actions). This file is the detailed record of *how* the
session got here and *why* each decision was made — read `HANDOFF.md` for
"what to do next," this file for "what happened and what the system
actually is."

---

## 1. What ACRR is

An AI-assisted Python code review and refactoring tool. Two HTTP endpoints
(`/review`, `/refactor`), a React frontend, and three genuinely different
kinds of "intelligence" layered on top of each other:

| Layer | What it does | Uses AI? |
|---|---|---|
| Deterministic engines | Style (`flake8`), security (`bandit`), 12 custom AST smell rules, rule-based refactor transforms | No |
| ML complexity model | Predicts Big-O for the dominant (worst) function in a file | No — RandomForest, not an LLM |
| AI layer | Deeper refactor suggestions + suspected correctness concerns | Yes — Groq / Gemini / Ollama |

This layering is deliberate, not incidental: deterministic engines carry
the parts that must be reliable and explainable every time; ML/AI is scoped
to the one job each is structurally suited for, and nothing else.

---

## 2. Architecture

### 2.1 Backend (`backend/app/`)

```
main.py                    Flask endpoints: /review, /refactor, /health
models.py                  Pydantic request/response schemas (single source of API contract)
parser.py                  Syntax gate — ast.parse() before anything else runs

review/
  __init__.py              run_all_checks() — coordinates the 4 checks below + ML + grade
  style.py                 flake8 wrapper
  complexity.py            radon cyclomatic complexity + maintainability index
  security.py              bandit wrapper
  smells.py                12 custom AST-based smell rules (SMELL001–012)

refactor/
  orchestrator.py          run_all_rules() — coordinates the rule-based transforms
  rules/                   dead_code, unused_vars, constant_fold, loop_opts, simplify_conditions
  diff.py                  unified diff generation
  ai_client.py             Groq / Gemini / Ollama provider abstraction

ml/
  features.py              24-feature structural extractor (the ONE source of truth
                            for training-time and prediction-time features)
  corpus.py                149 real, hand-labeled algorithms — training data as
                            actual Python source, never hand-typed feature vectors
  complexity_predictor.py  RandomForest training/inference, explain(), suggest_improvement()
  evaluate.py               Reports accuracy (stratified CV, grouped CV, benchmark)
  audit.py                  Adversarial audit — attacks the numbers rather than
                             producing them (leakage, sample size, selection bias,
                             calibration, robustness, label risk)
```

### 2.2 Frontend (`frontend/src/`)

```
App.jsx                    Two independent actions: "Review" and "Refactor"
api/client.js              Single fetch wrapper, both endpoints
components/
  CodeEditor.jsx            Code input
  ReviewPanel.jsx            Issues list, grouped by category
  ScoreCard.jsx               Grade + 5-way breakdown (style/complexity/security/
                               maintainability/big_o — big_o added this session)
  RefactorPanel.jsx           Diff view, applied rules, AI suggestions
```

### 2.3 Request flow

```
POST /review or /refactor
        |
   Syntax gate (ast.parse) ---- invalid ----> error response, nothing else runs
        |
   valid
        |
  +-----+------------------------------+
  |                                    |
/review                            /refactor
  |                                    |
4 checks in parallel              Rule-based transforms
(style, complexity,                    |
 security, smells)                ML complexity (before vs after)
  |                                    |
ML complexity                     AI layer (optional): Groq -> Gemini -> Ollama
(dominant function)                    |
  |                              suggestions + correctness_concerns
Quality score                          |
(style 20% + complexity 20%      Diff assembly
 + security 30% + maint 10%            |
 + big_o 20%, confidence-scaled) Response: diff + applied_rules +
  |                              ai_suggestions + correctness_concerns
Response: issues + grade
```

A full diagram of this flow was rendered earlier in the session (SVG,
inline) — this is the text equivalent for the written record.

### 2.4 The ML complexity model, specifically

- RandomForest classifier, 7 classes (O(1) through O(2^n)).
- Trained on 149 independent, hand-verified real algorithms (740→745
  augmented samples via structural no-op variants — renamed identifiers,
  added statements, added branches — that teach the model which features
  are irrelevant, not new evidence).
- 24 structural features (loop depth, recursion shape, sorting calls,
  linear-scan-in-loop, string-concat-in-loop, convergence-flag-while
  detection, etc.) — see `features.py`'s module docstring for the full
  numbered list and the reasoning behind each one.
- **Trust methodology**, established before this session and extended
  during it: contradiction checking (zero tolerance for the same feature
  vector mapping to two different labels), grouped cross-validation
  (variants of one algorithm never split across train/test), nested CV to
  detect hyperparameter-tuning optimism, and `audit.py` — a dedicated
  adversarial tool whose entire purpose is attacking the reported numbers.
- **Current honest numbers**: nested CV ~97.6%, 8/8 on genuinely unseen
  stress-test patterns, benchmark 23/23 (but 83% contaminated — the
  benchmark shares feature vectors with training data, kept as a smoke test
  only, never quoted as a generalization estimate).

---

## 3. Work log — what happened this session, in order

### 3.1 HANDOFF.md steps 1–5 (repo hygiene, corpus expansion)

1. **Commit/push** — the handoff doc claimed nothing was committed; that
   was stale. Repo was already clean; pushed one pending local commit.
   Hit and fixed a `git push` SSL failure (`CERTIFICATE_VERIFY_FAILED`) via
   `-c http.sslbackend=schannel` (a corporate-network TLS-inspection issue,
   Windows' own trust store has the cert, Python/git's bundled one doesn't
   — this exact root cause recurs later, see 3.3).
2. **Dead file removal** — already done before this session.
3. **Pinned dependencies** — `requirements-lock.txt` added via `pip freeze`.
4. **Full suite verification** — established baseline: 71 tests passing.
5. **Expanded the two thinnest complexity classes** — `O(n³+)` (9→17
   algorithms) and `O(n log n)` (13→21), all real Python source (Gaussian
   elimination, heap-based selection, k-way merges, brute-force
   longest-common-substring, etc.). Zero contradictions after expansion.
   Nested CV rose 96.8% → 97.8%.

*(Commits: `905d52f`, `4057b08` and earlier in this range.)*

### 3.2 Adversarial code review → 3-day bug-fix plan

User pasted a deliberately over-engineered "worst-known" Python solution to
a HackerRank problem, along with two AI-generated "analysis reports"
critiquing ACRR's review/refactor output. Rather than accept those reports
at face value, each claim was independently verified against the actual
code before acting — this became the working pattern for the rest of the
session (see 3.6).

**Day 1** (`fc18114`): Fixed two real, reproduced bugs.
- `_while_loop_halves` in `features.py` flagged ANY halving arithmetic
  anywhere inside a while-loop's subtree, with no check it was related to
  why the loop terminates — a deliberately irrelevant `x = x // 2` several
  statements deep triggered a confident but hallucinated "halves its range"
  explanation. Fixed by requiring the halved value to share a name with the
  loop's own test condition. Verified against all 10 real O(log n) corpus
  samples — all still detected correctly, false positive gone.
- Removed an unconditional "convert to a hash map" suggestion that fired on
  any nested loop regardless of whether a search/lookup pattern was
  actually present (confirmed it fired on a nested in-place swap-sort with
  no list being searched at all).

**Day 2** (`76227ed`): Started to "teach the model" that a specific
while-loop pattern was O(n²) — then proved that plan wrong before
implementing it. Verified empirically (2000 random trials) that an
early-exit comparison sort converges in ≤2 passes (true O(n²)), but a
structurally identical pattern — Bellman-Ford relaxation over an adjacency
matrix — genuinely needs up to n passes (true O(n³), also verified
empirically, passes growing with graph size). Same syntax, opposite ground
truth: no structural feature can resolve this. Pivoted to an honesty-first
fix: a new `has_convergence_flag_while` feature that adds an explicit
uncertainty caveat to the explanation instead of forcing a label either
way. Documented the ambiguity in `audit.py`'s `check_labels()` section
alongside its existing arguable-label entries (quicksort, BFS, gcd).

**Day 3** (`a760383`): Closed the two remaining scope-appropriate gaps.
- Wired Big-O into the quality grade (`review/__init__.py`) — previously
  the ML complexity prediction had zero influence on the letter grade shown
  next to it. Confidence-scaled term, 20% weight, falls back to the
  original 4-category weighting untouched if no prediction is available.
- `SMELL011` — inconsistent return type across branches (`return 0` vs
  `return "error"`), conservative literal-only check, no type inference.
- Extended the AI layer's system prompt and response schema to separately
  flag suspected correctness concerns, not just refactor suggestions — the
  architecturally honest fix for "the review missed that this doesn't
  solve the problem correctly," since only an LLM given the code can
  plausibly judge intent vs. implementation.

### 3.3 AI layer connectivity

Asked directly whether the ML model and AI layer actually worked — tested
both live rather than assuming.
- ML model: confirmed working immediately.
- AI layer: found `CERTIFICATE_VERIFY_FAILED` (same root cause as the
  earlier git SSL issue). Fixed with `pip-system-certs` (`ea3ccf5`).
- With SSL fixed, found the configured `GEMINI_API_KEY`'s project had
  `429 RESOURCE_EXHAUSTED`, `limit: 0` — a zero-quota project
  configuration, confirmed persistent by retrying after the API's own
  suggested delay. Not fixable from code.
- User independently tested a different Gemini key and hit a harder
  `403 PERMISSION_DENIED` (project access denied outright) — a different,
  more severe failure mode than the quota issue above. Pivoted to Groq,
  confirmed working via an independent test script.
- Wired Groq into `ai_client.py` as a third provider, tried first
  (`f5658d5`): Groq → Gemini → Ollama, falling through automatically.
  Gemini's code path left in place, untouched, for whenever that project's
  access gets resolved. Verified live end-to-end twice — a plain
  nested-loop sample and the original adversarial sample both returned
  real, validated suggestions and correctness concerns.

### 3.4 CodeBERT / CodeT5 evaluation

Asked to evaluate whether either model belonged in the project.
- Recommended against using either for classification or generation —
  duplicates the AI layer's job with a smaller, older model (CodeBERT:
  125M params, 2020; CodeT5-base: 220M, 2021, vs. a modern general-purpose
  LLM already integrated).
- Identified one legitimate exception: semantic near-duplicate function
  detection (extending `SMELL006`, which only catches exact-text
  duplicates). Chose CodeBERT over CodeT5 for this specific task
  (encoder-only fits an embedding task; CodeT5's encoder-decoder shape fits
  generation, which was already ruled out).
- Wrote a 3-day implementation plan, saved to
  `CODEBERT_DUPLICATE_DETECTION_PLAN.md`.

### 3.5 CodeBERT implementation, failure, and pivot

Directed to implement the plan. Day 1 was executed the same day:
- Installed `torch` (CPU wheel) + `transformers`, confirmed
  `microsoft/codebert-base` loads correctly (124.6M params).
- Built `duplicate_detector.py` (lazy-loaded singleton, mean-pooled
  embeddings, cosine similarity).
- **Calibration testing before committing to a threshold** caught the
  plan's core premise failing: raw embeddings scored every pair 0.97+ with
  no usable separation, and — worse — ranked a pair of semantically
  *different*, same-shaped functions (`get_name`/`get_email`-style getters)
  ABOVE a genuine near-duplicate. Tried `[CLS]`-token pooling instead of
  mean-pooling: same failure.
- Tried BERT-whitening (unsupervised, using `corpus.py`'s 149 algorithms as
  an unlabeled background set): separated "unrelated" from everything else,
  but the same backwards ranking on the getters case survived.
- Told to pick an option and proceed — chose the deterministic alternative
  over any CodeBERT variant. Uninstalled `torch`/`transformers` and their
  13 transitive dependencies (never added to `requirements.txt`).
- Built **`SMELL012`** (`fb2789e`): identifier normalization (Type-2 clone
  detection). Renames every locally-bound variable/parameter — including
  the function's own name, so recursive self-calls normalize correctly — to
  a canonical `VAR{n}` placeholder by order of first appearance, then
  compares `ast.dump()` of the normalized bodies for exact matches. Found
  and fixed two real bugs while building it: recursive self-calls weren't
  normalizing (function's own name wasn't in the rename mapping), and the
  statement-count threshold undercounted guard-clause functions
  (`len(node.body)` instead of a full recursive count). Verified correct on
  every calibration case that broke CodeBERT, including the backwards-
  ranking getters case.
- Asked what it would take to make CodeBERT work properly — investigated
  two existing HF Hub fine-tunes rather than theorize (`d89ce17`):
  `Lazyhope/python-clone-detection` requires `trust_remote_code=True`
  (arbitrary third-party code execution — declined without explicit
  sign-off); `mrm8488/codebert-finetuned-clone-detection` silently loads as
  a 100%-randomly-initialized model via the standard API (every weight
  `MISSING` in the load report, no error surfaced) and hard-crashes when
  loaded correctly as a classifier (undocumented custom 1536-dim
  architecture). Neither could even be evaluated for accuracy. This closes
  out all three CodeBERT-adjacent paths (raw embeddings, whitening,
  existing fine-tunes) as tried and documented, not just theorized about.

### 3.6 CodeT5+: verification harness, dataset, training, and integration

A later session picked up the CodeT5+ plan §3.6 referenced above (superseded
by `CODET5_PLAN.md`, gitignored/local-only, same treatment as
`CODEBERT_PLAN_V2.md`). Four parts, in order:

**Day 1 — differential verification harness** (`3a75805`, `4e174ef`,
`e4c6fd8`). `app/verify/sandbox.py`: cross-platform sandboxed execution —
`subprocess.Popen` + `communicate(timeout=...)` for hang protection
(reliable on Windows, unlike POSIX-only `resource.setrlimit`), a
`psutil`-based background thread summing RSS across the full process tree
for memory limits (a venv's `python.exe` launches the real interpreter as a
grandchild process on this machine — found and fixed as a real bug: the
naive version only watched the immediate child). `app/verify/differential.py`:
`verify_equivalent()` runs original vs. candidate on generated edge-case
inputs and compares outcomes (value or exception type, not message text);
`measure_speedup()` times both inside the sandboxed process (not from
outside — subprocess startup noise otherwise swamps genuine small
differences) and distinguishes a constant-factor speedup from an actual
complexity-class change. Wired into the existing Groq suggestion path
immediately (`e4c6fd8`) — every AI suggestion now carries `verified`/
`verification_note`, independent of whether CodeT5+ itself ever shipped.

**Day 2 — training-pair dataset** (`81fb74f`). `app/ml/t5/pairs.py`: 8
hand-written transformation families (`membership_to_set`, `index_to_dict`,
`string_concat_to_join`, `naive_fib_to_memo`, `pair_sum_count_to_hashmap`,
`sort_in_loop_to_once`, plus 2 held out entirely — `minmax_in_loop_to_running`,
`pop0_to_deque`), 25 instances each, both sides written by hand (never
LLM-generated — generating the "after" side with an LLM would mean
distilling, capping the model's ceiling at that LLM's floor). Held out by
whole family, not by row, so the final evaluation measures generalization,
not memorization of superficial variation. Every pair validated against
Day 1's harness before being trusted (200/200).

**Day 3 — fine-tuning, six real failures in sequence, each independently
diagnosed and fixed.** The notebook (`efdac3f`) itself needed environment
fixes first (wrong git branch defaulting away from `dev` — `66ae9a9`; wrong
Python interpreter resolved by a bare `"python"` call — `9616aae`; a
2023-era tokenizer data format newer libraries reject — `4813bbb`→`20a241d`,
after two version-pin guesses each traded one failure for a different one
before the actual fix — sanitizing the malformed data directly — held
regardless of library version; a `Trainer` kwarg rename — `2f8bdc3`). Then
three genuinely distinct *training* failures, in order: **undertraining**
(`345f0cd`) — 5 epochs at effective batch 32 worked out to ~25 total
optimizer steps, nowhere near enough to move a 220M-parameter model onto a
new prompt format, producing pure token-repetition garbage; fixed by
replacing the fixed epoch count with `load_best_model_at_end` +
`EarlyStoppingCallback` (`aa2ff6e`) so training length comes from validation
loss, not a guess. **FP16 instability** (`a55b02d`) — a `ValueError:
Attempting to unscale FP16 gradients`, a well-documented T5-family issue
distinct from generic mixed-precision problems; fixed by training in plain
FP32 (the free-tier T4 GPU doesn't support bf16, the usual alternative).
**NaN divergence in FP32** (`140bbc8`, `29f232b`) — a *different* T5-specific
instability (full-parameter AdamW fine-tuning without a properly-functioning
warmup), caught mid-run via a new `StopOnNaNLoss` callback and a post-training
NaN-weight check rather than discovered only after wasting a full evaluation
pass; fixed by an evidence-based learning-rate reduction (5e-5 → 2e-5 → 5e-6,
each step following the observed correlation between instability onset and
reaching the target learning rate, not a blind guess).

**Result**: `pass_rate: 0.34`, gate 1 (≥95% verification) **fails**, gate 2
(generalizes to ≥1 held-out family) **passes** — both held-out families
produced at least one genuinely correct, never-trained-on rewrite
(`pop0_to_deque` 15/25, `minmax_in_loop_to_running` 2/25). Per the plan,
this is technically a no-go for treating the model as a trustworthy default
feature — but a materially different no-go than every earlier run: this one
failed on *reliability*, not on being broken.

**Day 4 — integration, shipped anyway, deliberately reframed.** The plan's
95% bar was calibrated for a trusted-by-default feature. Given every
CodeT5+ output already passes back through Day 1's exact verification
harness before ever reaching a user — the same guarantee Groq suggestions
already carry — a low pass rate controls *how often the feature produces
anything*, not whether wrong code can reach the user. Shipped as an
explicitly experimental, off-by-default toggle on that basis:
`app/ml/t5/refactor_model.py` (lazy singleton, mirrors
`complexity_predictor.py`'s exact pattern for scikit-learn — `torch`/
`transformers` live in `requirements-optional.txt`, never `requirements.txt`,
and every failure path returns `None`, never raises), a thread-based soft
generation timeout, `/refactor`'s new `use_codet5` request field and
`codet5_available`/`codet5_suggestion` response fields, and a frontend
toggle disabled-with-a-tooltip until `/health` confirms availability (the
plan's own explicit UI requirement). A rejected candidate is still returned
(`verified: false`, its disagreement shown) rather than hidden, per the
plan: "show it, if at all, in a separate rejected section." Verified
end-to-end against the real fine-tuned weights (not just read/reasoned
about): a correct case shows `verified: true` plus a measured 44x speedup
with the complexity-class change correctly detected; a wrong case shows
`verified: false` with the exact disagreeing input and both outputs; the
main venv (no `torch` installed) confirmed to degrade to
`codet5_available: false` everywhere, including through the full live
browser flow, not only in isolated unit tests.

### 3.7 Working pattern established this session

Every pasted "analysis report" this session (the adversarial code review
critiques, the "100% robust model in 3 days" proposal, the Groq migration
report) contained at least one confident-sounding but incorrect technical
claim. Each was checked against the actual code or live behavior before
being acted on — a hallucinated while-loop explanation, an unverified
O(n³) vs. true O(n²) complexity claim (resolved by 2000-trial empirical
testing), an unrealistic "100% robust" framing (impossible for any ML
system, explained why), and a genuine-but-distinct SSL/quota diagnosis.
This verify-before-acting discipline is now recorded in persistent memory
so it carries into future sessions on this project.

---

## 4. Current state (as of the CodeT5+ integration commit — see §6)

| Area | Status |
|---|---|
| Test suite | **128/128 passing** (`pytest`), 23/23 (`verify_changes.py`) |
| Complexity model | 149 algorithms, 24 features, ~97.6% nested CV, fully audited |
| Review engine | 12 smell rules, Big-O feeds the grade, all deterministic checks working |
| Refactor engine | Rule-based transforms working; AI layer working via Groq; measured (not predicted) speed comparison on `/refactor` |
| AI layer | **Working end-to-end via Groq.** Gemini blocked at the account level (not fixable from code, documented). Ollama available but unconfigured. |
| CodeBERT | Evaluated, attempted, fully investigated, not used — deterministic `SMELL012` shipped instead |
| CodeT5+ | **Shipped as an experimental, off-by-default toggle.** 34% differential-verification pass rate, generalizes to both held-out families (gate 2 passes, gate 1 doesn't) — see §3.6. Every generation independently re-verified before being shown, same guarantee as Groq suggestions. |
| Deployment | **Not started, deliberately deferred.** Render/Vercel configs written (`render.yaml`, `backend/Procfile`, `backend/Dockerfile`), never executed |
| Documentation | `HANDOFF.md` (status/next-steps), `CODEBERT_DUPLICATE_DETECTION_PLAN.md` (full investigation record), this file. `CODET5_PLAN.md`, `CODET5_TRAINING_GUIDE.md`, `CODET5_MANUAL_CHECKLIST.md`, `STUDENT_GUIDE.md` are local-only (gitignored). |

## 5. Open items, in priority order

1. **Deployment** — the only remaining blocker to shipping live. Configs
   are ready; needs Render + Vercel accounts actually walked through. Note:
   `torch`/`transformers` (CodeT5+'s optional dependencies) add ~2.5-3.5 GB
   and are not installed by a plain `pip install -r requirements.txt` —
   deploying without `requirements-optional.txt` is the intended default,
   CodeT5+ simply reports unavailable.
2. **Improve CodeT5+'s pass rate**, if pursued further — the diagnosed
   weak spot is the "running accumulator" concept (`minmax_in_loop_to_running`
   at 8% vs. `pop0_to_deque` at 60%), not present in any currently-trained
   family. Not required to keep the current toggle shipped; it works today
   at its current accuracy because of the verification gate.
3. **Gemini account access** — not blocking (Groq covers the AI layer), but
   worth resolving eventually for redundancy. Requires the account owner's
   Google Cloud console, not fixable from this codebase.
4. Everything else from the original `HANDOFF.md` list is done.

## 6. Commit log, this session

```
<pending>  Wire CodeT5+ into /refactor as an opt-in, verification-gated toggle
29f232b  Lower T5 fine-tuning LR further (2e-5 -> 5e-6) based on a real trend
140bbc8  Fix T5 training divergence to NaN in FP32 (distinct from the FP16 issue)
eda6429  Keep CodeT5+ manual post-training checklist local-only
9393b06  Keep CodeT5+ training guide local-only
1379dcf  Add measured speed comparison to /refactor (original vs. refactored)
e16d369  Style AI-suggestion verification badges; keep planning docs local-only
aa2ff6e  Stop guessing training length: pick it from validation loss instead
345f0cd  Fix severe undertraining in T5 fine-tuning (~25 steps -> ~1,000+ steps)
a55b02d  Disable FP16 mixed precision in T5 fine-tuning (fixes gradient unscale crash)
2f8bdc3  Fix Seq2SeqTrainer tokenizer= removed in favor of processing_class=
20a241d  Fix the actual tokenizer bug (malformed data), revert version pin
14c99a7  Pin transformers version; use_fast=False alone did not fix tokenizer load
f6b6965  Add STUDENT_GUIDE.md to .gitignore
4813bbb  Fix notebook tokenizer load crashing on newer tokenizers library
66ae9a9  Fix notebook cloning the wrong branch, add a loud sanity check
9616aae  Fix notebook using bare 'python' instead of sys.executable for subprocess
efdac3f  Add Colab fine-tuning notebook and go/no-go gate (CodeT5+ plan, Day 3)
81fb74f  Add training-pair dataset for CodeT5+ fine-tuning (plan Day 2)
e4c6fd8  Wire behavioural verification into the Groq/Gemini AI suggestion path
4e174ef  Add differential testing harness (CodeT5+ plan, Day 1 continued)
3a75805  Add cross-platform execution sandbox (CodeT5+ plan, Day 1 foundation)
f892ee1  Add SESSION_REPORT.md — detailed architecture and work-log documentation
d89ce17  Document existing HF Hub clone-detection fine-tunes as also a dead end
fb2789e  Add SMELL012 (renamed-variable near-duplicate functions), retire CodeBERT
f5658d5  Add Groq as an AI provider, tried before Gemini
ea3ccf5  Fix AI layer SSL cert failure; document the separate quota block
a760383  Wire Big-O into the quality grade, add return-type smell, AI correctness flag
76227ed  Add convergence-flag detection for explanation honesty, not label-forcing
fc18114  Fix while_loop_halves false positive and unsubstantiated hash-map suggestion
4057b08  Expand thin O(n log n) and O(n3+) complexity classes
905d52f  Pin dependency versions, add handoff doc, refresh stale model fingerprint
```
