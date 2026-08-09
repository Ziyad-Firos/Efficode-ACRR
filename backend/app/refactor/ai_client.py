"""
refactor/ai_client.py — AI enhancement layer.

Calls Google Gemini (free tier) or a local Ollama model to get
deeper refactoring suggestions beyond what the rule engine can do.

Critical design decisions:
  1. NEVER blocks the response — if the AI call fails for any reason
     (timeout, no API key, rate limit, network error), the function
     returns ([], False) and the caller serves rule-based results only.

  2. EVERY suggested code snippet is run through ast.parse() before
     being included in the response. If it fails to parse, the
     suggestion is silently dropped. Broken code never reaches the user.

  3. The prompt asks for structured JSON output, not free-form prose.
     This makes responses programmatically parseable without regex hacks.

  4. API key is read from environment — never hardcoded. If the key
     is not set, the AI layer is skipped silently.

Environment variables:
  GEMINI_API_KEY     — Google Gemini API key (preferred)
  OLLAMA_BASE_URL    — e.g. http://localhost:11434 (for local Ollama)
  OLLAMA_MODEL       — model name, e.g. codellama (default: codellama)
  AI_TIMEOUT         — max seconds to wait for AI response (default: 15)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import List, Tuple

from app.models import AISuggestion
from app.parser import is_valid_python
from app.refactor.diff import make_diff

logger = logging.getLogger("acrr.refactor.ai_client")

_DEFAULT_TIMEOUT = int(os.getenv("AI_TIMEOUT", "15"))


# ── Prompt ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert Python code reviewer and refactoring assistant.
You will be given Python code that has already had basic rule-based optimizations applied.
Your job has two parts:

1. Suggest deeper refactoring improvements that rules cannot catch.
2. Separately, flag suspected LOGIC or CORRECTNESS problems — cases where the code
   looks like it's trying to do one thing but the implementation does something else,
   is missing an edge case its own structure implies it should handle, or returns
   inconsistent/wrong results for plausible inputs. This is the one thing only you can
   do here: rule-based checks and the ML complexity model can describe HOW code is
   shaped, never whether it does what it looks like it's meant to do.

Respond with ONLY a JSON object. No markdown, no explanation outside the object.
It must have exactly these two keys:
  "suggestions": array of refactoring suggestions, each an object with:
    "explanation": string — plain English description of the improvement
    "refactored_code": string — the complete improved Python code (must be valid Python)
  "correctness_concerns": array of strings — each a specific, concrete suspected logic
    bug, phrased so someone could go verify it (name the function, the condition, or
    the input that would expose it). Empty array if you see nothing suspicious.

Example response format:
{
  "suggestions": [
    {
      "explanation": "Replace O(n²) nested loop with a dictionary lookup for O(n) time complexity.",
      "refactored_code": "def find_pair(nums, target):\\n    seen = {}\\n    for i, n in enumerate(nums):\\n        if target - n in seen:\\n            return [seen[target - n], i]\\n        seen[n] = i\\n    return []"
    }
  ],
  "correctness_concerns": [
    "validate_age(n) returns True for n == -5 because it only checks 'n < 150', never 'n >= 0'."
  ]
}

Rules:
- Suggest at most 3 refactoring improvements.
- Only suggest refactors you are confident are correct and preserve behaviour.
- List at most 3 correctness concerns — the most concrete and verifiable ones, not
  vague style opinions. Do not restate something a linter would already catch
  (unused variables, magic numbers, etc.) as a correctness concern.
- If the code is already clean and correct, return empty arrays for both keys.
- Do not suggest changes, or flag concerns, that require knowing runtime values or
  external context you were not given.
"""


def _build_user_prompt(original: str, rule_refactored: str) -> str:
    return (
        f"Original code:\n```python\n{original}\n```\n\n"
        f"After rule-based refactoring:\n```python\n{rule_refactored}\n```\n\n"
        "Suggest further improvements as a JSON array."
    )


# ── Gemini client ─────────────────────────────────────────────────────────

async def _call_gemini(prompt: str, timeout: int) -> str:
    """Call Google Gemini API using google-genai SDK. Returns raw response text."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")

    loop = asyncio.get_event_loop()

    def _sync_call() -> str:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.3,
            ),
        )
        return response.text

    return await asyncio.wait_for(
        loop.run_in_executor(None, _sync_call),
        timeout=timeout,
    )


# ── Ollama client ─────────────────────────────────────────────────────────

async def _call_ollama(prompt: str, timeout: int) -> str:
    """Call a local Ollama instance. Returns raw response text."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model    = os.getenv("OLLAMA_MODEL", "codellama")

    try:
        import httpx  # type: ignore
    except ImportError:
        raise ImportError("httpx not installed. Run: pip install httpx")

    payload = {
        "model": model,
        "prompt": f"{_SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{base_url}/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")


# ── Response parser ───────────────────────────────────────────────────────

def _parse_suggestions(items, original_code: str) -> List[AISuggestion]:
    """Validate a raw suggestions array into AISuggestion objects. Suggestions
    with invalid Python (or missing required fields) are silently dropped."""
    suggestions: List[AISuggestion] = []
    if not isinstance(items, list):
        return suggestions

    for item in items[:3]:  # max 3 suggestions
        if not isinstance(item, dict):
            continue
        explanation = item.get("explanation", "").strip()
        code = item.get("refactored_code", "").strip()

        if not explanation or not code:
            continue

        # THE GUARDRAIL: validate the suggested code before accepting it
        valid = is_valid_python(code)
        if not valid:
            logger.info("AI suggestion dropped — invalid Python: %.80s…", code)

        diff = make_diff(original_code, code) if valid else ""

        suggestions.append(AISuggestion(
            explanation=explanation,
            diff=diff,
            validated=valid,
        ))

    return suggestions


def _parse_correctness_concerns(items) -> List[str]:
    """Validate a raw correctness_concerns array into a list of non-empty
    strings. Not code, so there is nothing to ast.parse — just filter junk."""
    if not isinstance(items, list):
        return []
    concerns = []
    for item in items[:3]:  # max 3, matches the prompt's own instruction
        if isinstance(item, str) and item.strip():
            concerns.append(item.strip())
    return concerns


def _parse_ai_response(raw: str, original_code: str) -> Tuple[List[AISuggestion], List[str]]:
    """
    Parse the AI's JSON response into (suggestions, correctness_concerns).

    Accepts the current object shape {"suggestions": [...], "correctness_concerns": [...]}.
    Also accepts a bare array — the old shape, and what a model might fall back to
    despite the system prompt — treating it as suggestions-only with no concerns,
    so a model that doesn't follow the updated schema still degrades usefully instead
    of losing every suggestion to a shape mismatch.
    """
    if not raw or not raw.strip():
        return [], []

    # Strip markdown code fences if the model added them despite instructions
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse AI JSON response: %s | raw: %.200s", exc, raw)
        return [], []

    if isinstance(parsed, list):
        return _parse_suggestions(parsed, original_code), []

    if isinstance(parsed, dict):
        suggestions = _parse_suggestions(parsed.get("suggestions", []), original_code)
        concerns = _parse_correctness_concerns(parsed.get("correctness_concerns", []))
        return suggestions, concerns

    logger.warning("AI response was neither a JSON array nor object — skipping")
    return [], []


# ── Public API ─────────────────────────────────────────────────────────────

async def get_ai_suggestions(
    original_code: str,
    rule_refactored_code: str,
) -> Tuple[List[AISuggestion], List[str], bool]:
    """
    Get AI-generated refactoring suggestions and suspected correctness concerns.

    Args:
        original_code:        The user's original code.
        rule_refactored_code: Code after rule-based refactoring.

    Returns:
        (suggestions, correctness_concerns, ai_available)
        suggestions:           List of AISuggestion objects (may be empty).
        correctness_concerns:  List of strings (may be empty). Always empty
                                when ai_available is False.
        ai_available:          True if the AI call succeeded (even with 0 of
                                either of the above).
    """
    timeout = _DEFAULT_TIMEOUT
    prompt  = _build_user_prompt(original_code, rule_refactored_code)

    # Try Gemini first, then Ollama, then gracefully degrade
    providers = []
    if os.getenv("GEMINI_API_KEY", "").strip():
        providers.append(("gemini", _call_gemini))
    if os.getenv("OLLAMA_BASE_URL", "").strip():
        providers.append(("ollama", _call_ollama))

    if not providers:
        logger.debug("No AI provider configured — skipping AI layer")
        return [], [], False

    for name, caller in providers:
        try:
            logger.info("Calling AI provider: %s", name)
            raw = await caller(prompt, timeout)
            suggestions, concerns = _parse_ai_response(raw, original_code)
            logger.info(
                "AI (%s): %d suggestion(s), %d correctness concern(s) returned",
                name, len(suggestions), len(concerns),
            )
            return suggestions, concerns, True

        except asyncio.TimeoutError:
            logger.warning("AI provider '%s' timed out after %ss", name, timeout)
        except (ImportError, ValueError) as exc:
            logger.warning("AI provider '%s' not available: %s", name, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI provider '%s' failed: %s", name, exc)

    # All providers failed
    return [], [], False
