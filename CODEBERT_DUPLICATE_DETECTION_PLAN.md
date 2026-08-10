# CodeBERT near-duplicate detection — attempted and abandoned

**Status: ABANDONED, replaced with a deterministic check.** This file
originally held a 3-day implementation plan (written 10 August 2026). Day 1
was executed the same day and the plan's core technical premise failed
empirical testing. Kept as a record of what was tried and why, so this
isn't re-attempted the same way later without knowing it already failed.

---

## What was tried

Per the original plan: `microsoft/codebert-base`, no fine-tuning, mean-pooled
embeddings, cosine similarity threshold. `torch` + `transformers` installed
(CPU-only wheel), the model downloaded and loaded successfully (124.6M
params, confirmed correct size), and a `duplicate_detector.py` module was
built with a lazy-loading singleton pattern matching `complexity_predictor.py`.

## Why it failed

Calibrated against hand-constructed test cases before committing to a
threshold — standard practice, and the same discipline this project's
other ML work has always used. Result:

```
--- RAW cosine similarity ---
1.0000  identical copy
0.9966  true near-duplicate (renamed vars, reordered)
0.9968  different, shaped-alike getters      <- ranked ABOVE the real duplicate
0.9704  unrelated functions
```

Every pair scored 0.97+, with no usable gap between "duplicate" and
"unrelated" — and the ranking was actively backwards: two semantically
*different* functions (`get_name`/`get_email`, same shape) scored higher
than a genuine near-duplicate pair. Tried `[CLS]`-token pooling instead of
mean-pooling — same failure, same backwards ranking (0.9990 vs 0.9981).

This is a known, documented property of raw pretrained BERT-family
embeddings called **anisotropy**: without a contrastive fine-tuning
objective, embeddings cluster in a narrow cone and cosine similarity stops
meaningfully separating anything. It's why the clone-detection literature
(BigCloneBench, POJ-104) fine-tunes CodeBERT for the task rather than using
it zero-shot — this plan's premise that pretrained embeddings would work
without fine-tuning was wrong.

**Also tried, also failed**: BERT-whitening (Su et al. 2021) — a
computationally cheap, unsupervised post-processing step, reusing
`corpus.py`'s 149 real algorithms as a background set (needs no labeled
pairs, unlike fine-tuning). Result:

```
--- WHITENED cosine similarity ---
1.0000  identical copy
0.2481  true near-duplicate
0.7843  different, shaped-alike getters      <- same backwards ranking
0.2143  unrelated functions
```

Whitening separated "unrelated" from everything else, but the same
backwards ranking survived. Likely because 149 background samples for a
768-dimensional embedding space is a 5x underdetermined covariance
estimate — but there's no way to know whether more background data would
actually fix the specific failure mode without testing it, and that's
exactly the kind of unresolved uncertainty this project doesn't ship on.

## Existing HF Hub fine-tunes — also checked, also a dead end

Investigated two candidates rather than leave this untested:

- **`Lazyhope/python-clone-detection`** (Python-specific, PoolC dataset) —
  requires `trust_remote_code=True` to load: executes arbitrary custom
  Python from a small community repo at model-load time. Stopped before
  running it; that's a real supply-chain trust decision, not something to
  wave through to save time on an experiment. No accuracy/F1 reported on
  the model card either, independent of the trust question.
- **`mrm8488/codebert-finetuned-clone-detection`** (BigCloneBench) — worse
  than expected. Loading it the "obvious" way (`AutoModel.from_pretrained`)
  *appeared* to succeed but silently returned a **100% randomly-initialized
  model** — every pretrained weight reported `MISSING` in the load report,
  which nothing surfaces as an error. Loading it correctly as a classifier
  (`AutoModelForSequenceClassification`) hard-crashed: `classifier.dense.weight`
  expects a 1536-dim input against the standard class's 768, meaning the
  real architecture concatenates two separately-encoded embeddings before
  classifying — a custom setup with no loadable code published for it.
  Not usable through any standard `transformers` call.

Neither candidate could even be evaluated for accuracy — both failed before
that question was reachable, for two different reasons (trust, architecture
mismatch). This closes out "check for an existing fine-tune first" as a
real, evidenced dead end, not just an untried option.

## What it would take to actually fix (documented, not pursued)

1. **More background data for whitening** — thousands of diverse functions
   (unlabeled, so cheaper than fine-tuning data) — untried, no guarantee.
2. **Fine-tune CodeBERT ourselves with a contrastive objective**, and
   critically, with **deliberate hard negatives** — same-shaped,
   semantically-different pairs like the getters case, since that's the
   exact failure mode observed twice. This is the literature-standard fix.
   Needs real labeled pairs, held-out grouped validation (same leakage
   discipline as `evaluate.py`/`audit.py`), and an adversarial calibration
   check before trusting it — a multi-week effort, not a 3-day one.

None of these were pursued. `torch`/`transformers` were uninstalled and
never made it into `requirements.txt`.

## What replaced it

`SMELL012` in `backend/app/review/smells.py` — deterministic identifier
normalization (Type-2 clone detection): rename every locally-bound
variable/parameter to a canonical `VAR{n}` placeholder by order of first
appearance, then compare `ast.dump()` of the normalized function bodies for
exact matches. Zero ML, zero training data, zero threshold to calibrate.
Verified against the exact same calibration cases that broke CodeBERT —
correct on all of them, including the backwards-ranking getters case.

Only catches renamed-variable duplicates with identical statement order
(true Type-2 clones), not reordered-statement duplicates — a smaller scope
than the original CodeBERT plan aimed for, delivered with certainty instead
of a plausible-sounding number that didn't hold up.
