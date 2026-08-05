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
Your job is to suggest deeper improvements that rules cannot catch.

Respond with ONLY a JSON array. No markdown, no explanation outside the array.
Each element must be an object with exactly these keys:
  "explanation": string — plain English description of the improvement
  "refactored_code": string — the complete improved Python code (must be valid Python)

Example response format:
[
  {
    "explanation": "Replace O(n²) nested loop with a dictionary lookup for O(n) time complexity.",
    "refactored_code": "def find_pair(nums, target):\\n    seen = {}\\n    for i, n in enumerate(nums):\\n        if target - n in seen:\\n            return [seen[target - n], i]\\n        seen[n] = i\\n    return []"
  }
]

Rules:
- Suggest at most 3 improvements.
- Only suggest changes you are confident are correct and preserve behaviour.
- If the code is already clean, return an empty array: []
- Do not suggest changes that require knowing runtime values or external context.
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

def _parse_ai_response(raw: str, original_code: str) -> List[AISuggestion]:
    """
    Parse the AI's JSON response into validated AISuggestion objects.
    Suggestions with invalid Python are silently dropped.
    """
    suggestions: List[AISuggestion] = []

    if not raw or not raw.strip():
        return suggestions

    # Strip markdown code fences if the model added them despite instructions
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        items = json.loads(cleaned)
        if not isinstance(items, list):
            logger.warning("AI response was not a JSON array — skipping")
            return suggestions
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse AI JSON response: %s | raw: %.200s", exc, raw)
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


# ── Public API ─────────────────────────────────────────────────────────────

async def get_ai_suggestions(
    original_code: str,
    rule_refactored_code: str,
) -> Tuple[List[AISuggestion], bool]:
    """
    Get AI-generated refactoring suggestions.

    Args:
        original_code:        The user's original code.
        rule_refactored_code: Code after rule-based refactoring.

    Returns:
        (suggestions, ai_available)
        suggestions:    List of AISuggestion objects (may be empty).
        ai_available:   True if the AI call succeeded (even if 0 suggestions).
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
        return [], False

    for name, caller in providers:
        try:
            logger.info("Calling AI provider: %s", name)
            raw = await caller(prompt, timeout)
            suggestions = _parse_ai_response(raw, original_code)
            logger.info("AI (%s): %d suggestion(s) returned", name, len(suggestions))
            return suggestions, True

        except asyncio.TimeoutError:
            logger.warning("AI provider '%s' timed out after %ss", name, timeout)
        except (ImportError, ValueError) as exc:
            logger.warning("AI provider '%s' not available: %s", name, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI provider '%s' failed: %s", name, exc)

    # All providers failed
    return [], False
