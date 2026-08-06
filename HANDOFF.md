# ACRR — handoff / current status

**Last updated:** 6 August 2026
**Branch:** `dev` (up to date with `origin/dev`)
**Ship target:** 12 August 2026

---

## Completion status

| Area | State | Remaining |
|---|---|---|
| Complexity engine (ML) | **Working, audited** | Nothing blocking — thin classes expanded (see below) |
| Refactor engine | **Working, tested** | Nothing blocking |
| Review engine | **Working, tested** | Big-O prediction does not feed the quality grade |
| Frontend | **Working, verified in browser** | Cosmetic only |
| Tests | **68 passing** + audit tool | Nothing blocking |
| Documentation | **Complete, honest numbers** | Nothing blocking |
| **Version control** | **Nothing committed** | ← **do this first** |
| **Deployment** | Config written, never run | Render + Vercel |
| Reproducibility | `>=` version ranges | Pin exact versions |

Roughly **80% complete**. The remaining 20% is operations — git, pinning,
deployment — not code. Every engine works and is tested.

### Measured accuracy (quote the first number)

- **Nested grouped cross-validation: 97.8%** — the honest figure (was 96.8%
  before step 5's corpus expansion; tuning optimism is now +0.0 pts)
- 8/8 on genuinely unseen patterns
- 23/23 on the held-out benchmark — **83% contaminated, do not quote this**
- 100% stratified CV — **meaningless, variants leak across folds**
- 148 independent algorithms (740 samples after augmentation — quote 148)
- Model is *under*-confident: mean confidence 85% vs 98% actual accuracy

---

## Do these in order

### 1. Commit — nothing is in version control yet

The git index still holds 154 files from v1, and 152 of them no longer exist
on disk. Not one current file is tracked. One bad edit loses everything.

```bash
git status
git add -A
git commit -m "Rebuild complexity engine, fix refactor correctness bugs, add ML audit"
git push origin dev
```

Before committing, confirm `backend/.env` is ignored:

```bash
git check-ignore -v backend/.env     # should print a .gitignore line
git log --all -- backend/.env        # should print nothing
```

It was never committed (verified by reading `.git/index`), so no key rotation
is needed — but re-check after staging.

### 2. Delete the dead file

`backend/app/middleware.py` is a no-op stub imported by nothing. Untracked, so
removing it is safe.

### 3. Pin dependency versions

`requirements.txt` uses `>=`. This already caused a discrepancy: scikit-learn
1.8 produced 98.1% grouped CV and 1.9 produced 97.4% on identical code. For a
result anyone can reproduce, pin exact versions:

```bash
cd backend
.venv\Scripts\activate
pip freeze > requirements-lock.txt
```

Keep `requirements.txt` with ranges for humans and `requirements-lock.txt` for
reproducing the reported numbers. Reference the lock file in the README's
accuracy section.

### 4. Verify the full suite once more

```bash
cd backend
python -m pytest tests -q         # expect 68 passed
python -m app.ml.evaluate         # grouped CV ~96%, benchmark 23/23
python -m app.ml.audit            # nested CV ~96.8%, 8/8 unseen
python verify_changes.py          # 23 end-to-end checks
```

`app.ml.audit` is the one that matters. It attacks the numbers rather than
producing them, and it has already caught a mislabelled benchmark case.

### 5. ~~Expand the two thin complexity classes~~ — done

Added 8 real algorithms to `O(n³+)` (Gaussian elimination, triangle counting,
Gram matrix, brute-force longest common substring, matrix chain DP,
brute-force max subarray sum, an all-`while` triple nest, plus one more) and
8 to `O(n log n)` (count inversions, heap sort, k-smallest/top-k-frequent via
heap, k-way merge, interval scheduling, rank queries, closest-pair-after-sort)
in `app/ml/corpus.py`, as Python source per the rule below. `O(n³+)` went
9 → 17 independent algorithms, `O(n log n)` 13 → 21.

Contradictory-vector check (same features, different label) came back **0**
across all 376 distinct feature vectors in the expanded corpus. Nested CV
rose 96.8% → **97.8%**, `O(n log n)` recall rose 85% → 96%, tuning optimism
dropped to +0.0 pts. `pytest` (71 passed) and `verify_changes.py` (23/23)
still pass. README and the numbers above are updated to match.

Rules that were followed when doing this:

- After adding samples, check for contradictory vectors (same features,
  different label). Zero is the target. A nonzero count names a missing
  feature — that check has already found three real gaps.
- If accuracy drops, that is information, not failure. It usually means the
  new samples exposed something the old corpus was not testing.

### 6. Deploy

`render.yaml`, `backend/Procfile` and `backend/Dockerfile` are ready.

- Backend to Render free tier. **Keep `--preload` in the gunicorn command** —
  it trains the model once in the parent process before workers fork.
- Set `ALLOWED_ORIGINS` to the deployed frontend URL. Unset, the backend
  accepts any origin and logs a warning at startup.
- Set `GEMINI_API_KEY` in the Render dashboard, never in the repo.
- Frontend to Vercel; set `VITE_API_BASE_URL` to the Render URL.
- Watch memory — scikit-learn plus a 150-tree forest on a 512 MB instance is
  not free. If it is tight, drop `n_estimators` and re-run `evaluate.py` to
  confirm accuracy holds.

### 7. Optional — wire Big-O into the quality grade

Currently the review grade uses only radon's *cyclomatic* complexity, which
reads 100/100 for a clean nested loop. Messy `O(n²)` demo code scores B (87).
The app's headline feature does not influence its own grade. This is a design
decision, not a bug — decide deliberately.

---

## Things not to undo

These were each fixed after a verified failure. Reverting any of them
reintroduces a real bug.

- **`simplify_conditions` does not rewrite `x == True` → `x`** unless the left
  side is provably a bool. `2 == True` is `False` but `if 2` is truthy, so the
  rewrite silently changes behaviour. It is reported as advice instead
  (`applied: false`).
- **`loop_to_comprehension` refuses three cases**: accumulator read inside the
  appended expression, loop variable used after the loop, loop target
  shadowing the accumulator. All three produced code that raises `NameError`.
- **`sorted`/`sort` are excluded from `bulk_collection_op`** in `features.py`.
  Including them made the feature ambiguous between linear and n-log-n work
  and cost nine `O(n log n)` samples.
- **`test_refactor_preserves_behaviour` executes** the original and refactored
  code and compares results. Structural assertions cannot catch a refactor
  that produces plausible-looking broken output. It found three real bugs on
  its first run.
- **The stress set in `audit.py` decays.** Fixing a failure means adding that
  pattern to the corpus, which contaminates that case. The audit splits
  "unseen" from "regression" and warns below 8 clean cases. Refresh it with
  new patterns rather than watching the number climb.

---

## Known limitations (already documented in the README)

- No interprocedural analysis — a helper's cost is not propagated to its caller
- Amortised costs not modelled (`list.append` treated as O(1))
- Input-dependent complexity reports the typical case (quicksort → n log n)
- Space complexity not predicted at all
- Labels encode one opinion; `n` is not consistently defined across classes
  (magnitude for `gcd`, element count for sorting, V+E for BFS)
