"""
main.py — Flask application. The ONLY entry point.

Run with:
    python -m app.main
  or
    flask --app app.main run --port 8000 --debug
"""

from __future__ import annotations

import asyncio
import logging
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from app.parser import parse_code
from app.review import run_all_checks
from app.refactor.orchestrator import run_all_rules
from app.refactor.diff import make_diff
from app.refactor.ai_client import get_ai_suggestions
from app.ml.complexity_predictor import predict_complexity
from app.models import OptimizationLevel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("acrr.main")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# CORS: allow only the deployed frontend in production.
# `CORS(app)` with no arguments allows every origin, which is fine on
# localhost and wrong once this is on the public internet. Set
# ALLOWED_ORIGINS to a comma-separated list; the permissive default is kept
# for local development so nothing breaks when the variable is unset.
_allowed = os.getenv("ALLOWED_ORIGINS", "").strip()
if _allowed:
    CORS(app, origins=[o.strip() for o in _allowed.split(",") if o.strip()])
    logger.info("CORS restricted to: %s", _allowed)
else:
    CORS(app)
    logger.warning(
        "ALLOWED_ORIGINS is not set — accepting requests from any origin. "
        "Set it before deploying publicly."
    )
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024  # 200 KB hard limit — Flask returns 413 automatically

# Warm the complexity model at import time.
#
# The model trains in well under a second, but doing that lazily means the
# FIRST user request pays for it — and on a free-tier host that sleeps, that
# request is already paying for container wake-up. Training here moves the
# cost to startup, where nobody is waiting on it.
try:
    from app.ml.complexity_predictor import _get_model

    if _get_model() is not None:
        logger.info("Complexity model ready.")
    else:
        logger.warning(
            "Complexity model unavailable — scikit-learn is not installed. "
            "/refactor will return complexity: null."
        )
except Exception as exc:  # noqa: BLE001
    logger.warning("Could not warm the complexity model: %s", exc)

logger.info("ACRR backend ready.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _analyser_status() -> dict:
    """
    Which analysers are actually usable right now.

    Every analyser fails silently by design — a missing tool returns an empty
    issue list rather than an error. That is right for robustness and wrong
    for confidence: a review can come back clean simply because nothing ran.
    Reporting availability here makes a missing analyser visible instead.
    """
    import shutil

    def module_present(name: str) -> bool:
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    return {
        "style_flake8": shutil.which("flake8") is not None,
        "security_bandit": shutil.which("bandit") is not None,
        "complexity_radon": module_present("radon"),
        "smells_ast": True,                       # stdlib only, always available
        "ml_sklearn": module_present("sklearn"),
    }


@app.get("/health")
def health():
    """Liveness check plus analyser availability."""
    ai_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("OLLAMA_BASE_URL", "")
    analysers = _analyser_status()
    degraded = [name for name, ok in analysers.items() if not ok]

    return jsonify({
        "status": "ok",
        "ai_configured": bool(ai_key),
        "version": "1.0.0",
        "analysers": analysers,
        "degraded": degraded,
    })


@app.post("/review")
def review():
    """
    POST /review
    Body: { "code": "..." }

    Full code review pipeline:
      1. Syntax check
      2. Style (flake8), Complexity (radon), Security (bandit), Smells (AST)
      3. Quality score A–F
    """
    data = request.get_json(silent=True) or {}
    code = data.get("code")
    if code is None or not isinstance(code, str):
        return jsonify({"error": "Field 'code' is required and must be a string"}), 400
    code = code.strip()

    if not code:
        return jsonify({"error": "Missing required field: code"}), 400
    if len(code) > 100_000:
        return jsonify({"error": "Code too long (max 100 000 chars)"}), 400

    logger.info("POST /review  code_len=%d", len(code))

    # Syntax gate
    parse_result = parse_code(code)
    if not parse_result.valid:
        return jsonify({
            "valid": False,
            "syntax_error": parse_result.error,
            "issues": [],
            "quality_score": None,
            "summary": f"Syntax error: {parse_result.error}",
        })

    # Run all checks (async inside sync Flask — use asyncio.run)
    result = asyncio.run(run_all_checks(code))
    return jsonify(result.model_dump())


@app.post("/refactor")
def refactor():
    """
    POST /refactor
    Body: { "code": "...", "level": "medium", "use_ai": true }

    Full refactor pipeline:
      1. Syntax gate
      2. Rule-based transformations
      3. AI enhancement (optional)
      4. Complexity prediction (skipped if sklearn unavailable)
      5. Diff generation
    """
    data = request.get_json(silent=True) or {}
    code = data.get("code")
    if code is None or not isinstance(code, str):
        return jsonify({"error": "Field 'code' is required and must be a string"}), 400
    code = code.strip()
    level_str = data.get("level", "medium")
    use_ai = bool(data.get("use_ai", True))

    if not code:
        return jsonify({"error": "Missing required field: code"}), 400
    if len(code) > 100_000:
        return jsonify({"error": "Code too long (max 100 000 chars)"}), 400

    # Validate level
    try:
        level = OptimizationLevel(level_str)
    except ValueError:
        level = OptimizationLevel.MEDIUM

    logger.info("POST /refactor  code_len=%d  level=%s  use_ai=%s", len(code), level, use_ai)

    # Syntax gate
    parse_result = parse_code(code)
    if not parse_result.valid:
        return jsonify({
            "original_code": code,
            "refactored_code": code,
            "diff": "",
            "applied_rules": [],
            "ai_suggestions": [],
            "ai_available": False,
            "complexity": None,
            "summary": f"Cannot refactor: syntax error — {parse_result.error}",
        })

    # Rule-based refactoring
    rule_result = run_all_rules(code, level=level)

    # AI layer (optional)
    ai_suggestions = []
    ai_available = False
    if use_ai:
        ai_suggestions, ai_available = asyncio.run(get_ai_suggestions(
            original_code=code,
            rule_refactored_code=rule_result.refactored_code,
        ))

    # Complexity prediction
    complexity = predict_complexity(code, rule_result.refactored_code)

    # Diff
    diff = make_diff(code, rule_result.refactored_code)

    # Summary
    # Only count rules that actually modified the code. Advisory entries
    # (applied=False) are still returned so the UI can show them, but claiming
    # "3 changes applied" when one was only a suggestion is misleading.
    changes_count = len([r for r in rule_result.applied_rules if r.applied])
    suggestion_count = len([r for r in rule_result.applied_rules if not r.applied])
    ai_count = len([s for s in ai_suggestions if s.validated])
    summary_parts = []
    if changes_count:
        summary_parts.append(
            f"{changes_count} rule-based change{'s' if changes_count != 1 else ''} applied"
        )
    if ai_count:
        summary_parts.append(
            f"{ai_count} AI suggestion{'s' if ai_count != 1 else ''} available"
        )
    if suggestion_count:
        summary_parts.append(
            f"{suggestion_count} suggestion{'s' if suggestion_count != 1 else ''} "
            f"not auto-applied"
        )
    if not summary_parts:
        summary_parts = ["No changes needed — code looks clean"]

    return jsonify({
        "original_code": code,
        "refactored_code": rule_result.refactored_code,
        "diff": diff,
        "applied_rules": [r.model_dump() for r in rule_result.applied_rules],
        "ai_suggestions": [s.model_dump() for s in ai_suggestions],
        "ai_available": ai_available,
        "complexity": complexity.model_dump() if complexity else None,
        "summary": ", ".join(summary_parts) + ".",
    })


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    """
    Preserve real HTTP status codes.

    This handler must exist and must come before the catch-all below.
    Flask dispatches errorhandler(Exception) for HTTPException too, so
    without this a 404, a 405, or a 413 (payload over MAX_CONTENT_LENGTH)
    was being rewritten into a 500 with a misleading "internal error"
    message — the client could not tell a bad request from a server fault.
    """
    return jsonify({
        "error": e.description or e.name,
        "status": e.code,
    }), (e.code or 500)


@app.errorhandler(Exception)
def handle_exception(e):
    """Catch-all for genuine server faults only."""
    logger.exception("Unhandled exception")
    return jsonify({"error": "An internal error occurred. Please try again."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
