"""
ml/t5/refactor_model.py — optional local CodeT5+ refactoring model.

Mirrors the exact lazy-singleton / degrade-cleanly pattern already
established by app/ml/complexity_predictor.py for scikit-learn: torch and
transformers are OPTIONAL dependencies (requirements-optional.txt, never
requirements.txt), the model is loaded once on first use rather than at
import time, and every failure path here returns None instead of raising --
a plain `pip install -r requirements.txt` with no torch installed, or no
model files downloaded, must produce a perfectly working app with this
feature simply unavailable.

Status as of this module's introduction: the fine-tuned model passes the
plan's gate 2 (generalizes to >=1 held-out transformation family) but not
gate 1 (>=95% differential verification -- actual figure 34%). Per the
plan, that is an overall NO-GO for treating this as a trustworthy default
feature. It is wired in anyway, deliberately, as an opt-in, clearly-labelled
experimental toggle: every generation is run back through the SAME
verify_equivalent() harness used for Groq's AI suggestions before it is ever
shown, so a generation that is wrong is filtered out before reaching the
user, not shown as a trustworthy answer. The 34% figure controls how often
the toggle produces anything at all, not whether wrong code can reach the
user -- that distinction is the reason this ships now instead of waiting for
gate 1 to pass. See CODET5_PLAN.md / CODET5_MANUAL_CHECKLIST.md (repo root,
gitignored) for the full reasoning.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Optional

logger = logging.getLogger("acrr.ml.t5.refactor_model")

# backend/models/codet5p_finetuned/ -- same top-level backend/models/
# convention already used for other large downloaded model caches
# (flan_t5_cache/, flan_t5_fine_tuned/ in .gitignore), distinct from the
# small RandomForest pickle's app/ml/models/. The weights themselves are
# never committed -- only this loader code is.
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "codet5p_finetuned"

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512
GENERATION_TIMEOUT_SECONDS = 20.0  # generous vs. the ~3s measured locally --
# covers a cold/slow CPU without letting a stuck generation hang a request

_PROMPT_PREFIX = "optimize:\n"  # must exactly match training's format_input()


def _format_input(code: str) -> str:
    return f"{_PROMPT_PREFIX}{code}"


def _load_model():
    """
    Load the tokenizer + model once. Returns (tokenizer, model) or
    (None, None) on any failure -- missing libraries, missing weights, or a
    corrupt/incompatible checkpoint are all just "unavailable", never a
    crash. Logged at the level that matches how expected the failure is:
    missing torch/transformers (the common case, since they're optional) is
    an info-level "not installed", a present-but-broken model directory is a
    warning, since that means someone intended this to work.
    """
    if not _MODEL_DIR.exists():
        logger.info("CodeT5+ model directory not found at %s — feature disabled.", _MODEL_DIR)
        return None, None

    try:
        import torch  # noqa: F401  -- imported for its side effect of registering backends
        from transformers import AutoTokenizer, T5ForConditionalGeneration
    except ImportError:
        logger.info(
            "torch/transformers not installed — CodeT5+ feature disabled. "
            "Install with: pip install -r requirements-optional.txt"
        )
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
        model = T5ForConditionalGeneration.from_pretrained(str(_MODEL_DIR))
        model.eval()
    except Exception as exc:  # noqa: BLE001 -- a broken checkpoint must degrade, not crash the app
        logger.warning("CodeT5+ model directory present but failed to load: %s", exc)
        return None, None

    logger.info("CodeT5+ model loaded from %s", _MODEL_DIR)
    return tokenizer, model


_tokenizer = None
_model = None
_load_attempted = False


def _get_model():
    global _tokenizer, _model, _load_attempted
    if not _load_attempted:
        _load_attempted = True
        _tokenizer, _model = _load_model()
    return _tokenizer, _model


def codet5_available() -> bool:
    """
    Cheap availability check for /health and the request-time gate in
    main.py — does NOT load the model itself just to answer this (that
    would defeat the point of lazy loading on a health-check endpoint that
    might be polled often), but DOES check both halves of what "available"
    actually needs: the weights are present, AND torch/transformers are
    importable. Checking only the directory would report "available" even
    in an environment with no torch installed, where every real
    generate_refactor() call would then silently return None — a
    misleading signal for a toggle whose whole job is telling the user
    honestly whether this will do anything. If the weights are present but
    corrupt, that's still only discovered on first real load, same
    tradeoff complexity_predictor.py accepts for the RandomForest cache.
    """
    if not _MODEL_DIR.exists():
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _generate(tokenizer, model, code: str, num_beams: int = 4) -> str:
    inputs = tokenizer(
        _format_input(code), return_tensors="pt",
        max_length=MAX_SOURCE_LENGTH, truncation=True,
    )
    import torch
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_length=MAX_TARGET_LENGTH, num_beams=num_beams,
            no_repeat_ngram_size=3, early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="codet5-generate")


def generate_refactor(code: str) -> Optional[str]:
    """
    Generate a candidate refactoring for `code`, or None if the model is
    unavailable, generation fails, or it doesn't finish within
    GENERATION_TIMEOUT_SECONDS. Never raises.

    The timeout is enforced by not waiting past it, not by killing the
    underlying computation (Python can't safely interrupt a running
    torch call) -- an orphaned generation is left to finish in its worker
    thread and its result is simply discarded. That's an acceptable
    tradeoff for a single-worker local model: it wastes some CPU/GPU time
    on the rare timeout, but never blocks the request thread past the
    limit, which is the actual guarantee callers need.
    """
    tokenizer, model = _get_model()
    if tokenizer is None or model is None:
        return None

    future = _executor.submit(_generate, tokenizer, model, code)
    try:
        return future.result(timeout=GENERATION_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        logger.warning("CodeT5+ generation exceeded %.0fs — returning unavailable for this request.",
                        GENERATION_TIMEOUT_SECONDS)
        return None
    except Exception as exc:  # noqa: BLE001 -- generation failing must never break the response
        logger.warning("CodeT5+ generation failed: %s", exc)
        return None
