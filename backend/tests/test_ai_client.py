"""
tests/test_ai_client.py — AI response parsing correctness.

Why this file exists
--------------------
The AI layer's response schema changed from a bare JSON array of suggestions
to an object with two keys ("suggestions" and "correctness_concerns"), so the
system prompt could ask the model to separately flag suspected logic bugs —
something only an LLM given the code can plausibly judge; the rule engine and
ML complexity model only ever describe HOW code is shaped, never whether it
does what it looks like it's meant to do.

None of this needs a live API key: _parse_ai_response is a pure function
over the raw text a provider would have returned, so it's tested directly.
get_ai_suggestions() itself is checked only for its no-provider-configured
degrade path, which is also deterministic and key-free.

Runs under pytest, or standalone with `python tests/test_ai_client.py`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.refactor.ai_client import (   # noqa: E402
    _parse_ai_response,
    _verify_suggestion,
    get_ai_suggestions,
)


# ---------------------------------------------------------------------------
# _parse_ai_response — current object shape
# ---------------------------------------------------------------------------

def test_parses_suggestions_and_concerns():
    raw = json.dumps({
        "suggestions": [
            {
                "explanation": "Use a dict for O(1) lookups.",
                "refactored_code": "def f(x):\n    return x\n",
            }
        ],
        "correctness_concerns": [
            "validate(n) never checks n >= 0, so negative ages pass silently.",
        ],
    })
    suggestions, concerns = _parse_ai_response(raw, original_code="def f(x): return x\n")
    assert len(suggestions) == 1
    assert suggestions[0].validated is True
    assert concerns == ["validate(n) never checks n >= 0, so negative ages pass silently."]


def test_invalid_python_suggestion_is_flagged_but_concerns_survive():
    """Invalid Python isn't dropped from the list -- it's kept with
    validated=False and an empty diff, so it never reaches the user as
    appliable code (see the guardrail in _parse_suggestions) but its
    presence is still visible. Independent of that, correctness_concerns
    parses from its own key and isn't affected by a bad suggestion."""
    raw = json.dumps({
        "suggestions": [
            {"explanation": "broken", "refactored_code": "def f(:\n    pass"},
        ],
        "correctness_concerns": ["something looks off in g()"],
    })
    suggestions, concerns = _parse_ai_response(raw, original_code="def f(): pass\n")
    assert len(suggestions) == 1
    assert suggestions[0].validated is False
    assert suggestions[0].diff == ""
    assert concerns == ["something looks off in g()"]


def test_concerns_truncated_to_three():
    raw = json.dumps({
        "suggestions": [],
        "correctness_concerns": ["a", "b", "c", "d", "e"],
    })
    _, concerns = _parse_ai_response(raw, original_code="")
    assert concerns == ["a", "b", "c"]


def test_non_string_and_blank_concerns_filtered():
    raw = json.dumps({
        "suggestions": [],
        "correctness_concerns": ["real concern", "", "   ", 42, None],
    })
    _, concerns = _parse_ai_response(raw, original_code="")
    assert concerns == ["real concern"]


# ---------------------------------------------------------------------------
# _parse_ai_response — backward compatibility / malformed input
# ---------------------------------------------------------------------------

def test_bare_array_still_parses_as_suggestions_only():
    """A model that ignores the updated schema and returns the old bare-array
    shape should still yield its suggestions, just with no concerns -- not
    lose everything to a shape mismatch."""
    raw = json.dumps([
        {"explanation": "tidy this up", "refactored_code": "def f():\n    return 1\n"},
    ])
    suggestions, concerns = _parse_ai_response(raw, original_code="def f(): return 1\n")
    assert len(suggestions) == 1
    assert concerns == []


def test_empty_string_returns_nothing():
    assert _parse_ai_response("", original_code="") == ([], [])


def test_invalid_json_returns_nothing():
    assert _parse_ai_response("not json at all {{{", original_code="") == ([], [])


def test_markdown_fenced_json_still_parses():
    raw = "```json\n" + json.dumps({
        "suggestions": [],
        "correctness_concerns": ["flagged despite fences"],
    }) + "\n```"
    _, concerns = _parse_ai_response(raw, original_code="")
    assert concerns == ["flagged despite fences"]


# ---------------------------------------------------------------------------
# _verify_suggestion — behavioural verification of an AI suggestion
# ---------------------------------------------------------------------------
# These spawn real subprocesses via app/verify -- no API key needed, since
# they test the verification step on hand-written original/candidate pairs,
# not a live model response. Slower than pure-function tests for the same
# reason tests/test_sandbox.py and tests/test_differential.py are.

def test_verify_suggestion_confirms_correct_refactoring():
    original = (
        "def common_elements(list_a, list_b):\n"
        "    result = []\n"
        "    for x in list_a:\n"
        "        if x in list_b:\n"
        "            result.append(x)\n"
        "    return result\n"
    )
    candidate = (
        "def common_elements(list_a, list_b):\n"
        "    lookup = set(list_b)\n"
        "    return [x for x in list_a if x in lookup]\n"
    )
    verified, note = _verify_suggestion(original, candidate)
    assert verified is True
    assert "agreed" in note


def test_verify_suggestion_rejects_wrong_refactoring():
    """A suggestion that parses fine but changes behaviour must come back
    verified=False, not just silently accepted because it's valid Python."""
    original = (
        "def find_pair(nums, target):\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(len(nums)):\n"
        "            if i != j and nums[i] + nums[j] == target:\n"
        "                return (i, j)\n"
        "    return None\n"
    )
    wrong_candidate = (
        "def find_pair(nums, target):\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        complement = target - n\n"
        "        if complement in seen:\n"
        "            return (seen[complement], i)\n"
        "        seen[n] = i + 1\n"  # bug: off by one
        "    return None\n"
    )
    verified, note = _verify_suggestion(original, wrong_candidate)
    assert verified is False
    assert "disagreed" in note


def test_verify_suggestion_skips_multi_function_original():
    """Ambiguous which function the suggestion is refactoring -- must
    decline to guess, not silently pick one."""
    original = "def f():\n    return 1\n\ndef g():\n    return 2\n"
    candidate = "def f():\n    return 1\n\ndef g():\n    return 2\n"
    verified, note = _verify_suggestion(original, candidate)
    assert verified is None
    assert "found 2" in note


def test_verify_suggestion_skips_renamed_function():
    original = "def add(a, b):\n    return a + b\n"
    candidate = "def sum_two(a, b):\n    return a + b\n"  # renamed
    verified, note = _verify_suggestion(original, candidate)
    assert verified is None
    assert "does not define" in note


# ---------------------------------------------------------------------------
# get_ai_suggestions — degrade path (no network, no key required)
# ---------------------------------------------------------------------------

def test_no_provider_configured_degrades_cleanly(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    suggestions, concerns, available = asyncio.run(
        get_ai_suggestions(original_code="def f(): return 1\n", rule_refactored_code="def f(): return 1\n")
    )
    assert suggestions == []
    assert concerns == []
    assert available is False


def test_groq_tried_before_gemini(monkeypatch):
    """Groq is the currently-verified-working provider (see the module
    docstring) -- it must be tried first when both keys are configured, not
    just whichever happens to be listed first in the environment."""
    import app.refactor.ai_client as ai_client

    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    called = []

    async def fake_groq(prompt, timeout):
        called.append("groq")
        return json.dumps({"suggestions": [], "correctness_concerns": ["from groq"]})

    async def fake_gemini(prompt, timeout):
        called.append("gemini")
        return json.dumps({"suggestions": [], "correctness_concerns": ["from gemini"]})

    monkeypatch.setattr(ai_client, "_call_groq", fake_groq)
    monkeypatch.setattr(ai_client, "_call_gemini", fake_gemini)

    suggestions, concerns, available = asyncio.run(
        get_ai_suggestions(original_code="def f(): return 1\n", rule_refactored_code="def f(): return 1\n")
    )
    assert available is True
    assert concerns == ["from groq"]
    assert called == ["groq"]  # gemini never called -- groq succeeded first


def test_falls_through_to_gemini_when_groq_fails(monkeypatch):
    import app.refactor.ai_client as ai_client

    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    async def failing_groq(prompt, timeout):
        raise ValueError("quota exceeded")

    async def working_gemini(prompt, timeout):
        return json.dumps({"suggestions": [], "correctness_concerns": ["from gemini"]})

    monkeypatch.setattr(ai_client, "_call_groq", failing_groq)
    monkeypatch.setattr(ai_client, "_call_gemini", working_gemini)

    suggestions, concerns, available = asyncio.run(
        get_ai_suggestions(original_code="def f(): return 1\n", rule_refactored_code="def f(): return 1\n")
    )
    assert available is True
    assert concerns == ["from gemini"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
