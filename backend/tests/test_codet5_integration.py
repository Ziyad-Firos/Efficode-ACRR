"""
tests/test_codet5_integration.py — the local CodeT5+ toggle degrades cleanly.

Why this file exists
--------------------
The one guarantee that must never break, regardless of anything else about
this feature: a plain `pip install -r requirements.txt` (no torch, no
transformers, no downloaded model weights) must produce a perfectly working
app, with app.ml.t5.refactor_model simply reporting itself unavailable.
This is the environment these tests actually run in -- torch/transformers
are deliberately NOT in requirements.txt (see requirements-optional.txt),
so a normal `pytest` run here IS the real degrade-cleanly check, not a
simulation of one.

A test that actually loads the fine-tuned model and checks generation
quality would need the real (536 MB, gitignored, never committed) weights
present, which most environments running this suite won't have -- that
verification was done manually against real weights while building this
feature (see CODET5_MANUAL_CHECKLIST.md, gitignored) and isn't repeated
here as an automated test.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app  # noqa: E402
from app.ml.t5.refactor_model import codet5_available, generate_refactor  # noqa: E402


def test_codet5_unavailable_without_torch():
    """
    This suite runs in an environment with no torch/transformers installed
    on purpose (see the module docstring). If this assertion ever fails, it
    means torch/transformers ended up in requirements.txt by accident --
    exactly the mistake this feature was built to avoid.
    """
    assert codet5_available() is False


def test_generate_refactor_returns_none_without_torch():
    result = generate_refactor("def f(x):\n    return x\n")
    assert result is None


def test_health_reports_codet5_available_false():
    client = app.test_client()
    data = client.get("/health").get_json()
    assert "codet5_available" in data
    assert data["codet5_available"] is False


def test_refactor_with_use_codet5_degrades_cleanly():
    """The request must succeed normally -- use_codet5=True must never
    cause an error, timeout, or missing field, only an honest
    codet5_suggestion: null."""
    client = app.test_client()
    code = "def f(x):\n    return x + 1\n"
    resp = client.post("/refactor", json={"code": code, "use_ai": False, "use_codet5": True})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["codet5_available"] is False
    assert data["codet5_suggestion"] is None
    # every other field must still be present and normal -- this feature
    # being off must not have any effect on the rest of the response.
    # code.strip() because /refactor strips the input before echoing it
    # back, same as every other endpoint in this file.
    assert data["original_code"] == code.strip()
    assert "refactored_code" in data


def test_refactor_syntax_error_path_has_consistent_codet5_shape():
    client = app.test_client()
    resp = client.post("/refactor", json={"code": "def f(:\n", "use_ai": False})
    data = resp.get_json()
    assert data["codet5_suggestion"] is None
    assert "codet5_available" in data
