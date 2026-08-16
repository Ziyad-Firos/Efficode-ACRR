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

What happens when this file's own precondition doesn't hold
-------------------------------------------------------------
If torch/transformers (and the model weights) ARE installed in whatever
environment runs this suite -- e.g. a dev machine that ran `pip install -r
requirements-optional.txt` to try the toggle locally, which is a real,
supported, EXPECTED thing to do, not a mistake -- the four
"without torch" tests below skip themselves rather than fail. A skip
here means exactly what "verified: None" means elsewhere in this
project: the question wasn't asked in this environment, not that the
answer is wrong. Caught for real: these hard-failed the first time this
file ran on a machine that had genuinely installed the optional deps,
which is precisely the scenario this paragraph now handles on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app  # noqa: E402
from app.ml.t5.refactor_model import codet5_available, generate_refactor  # noqa: E402

_skip_if_torch_installed = pytest.mark.skipif(
    codet5_available(),
    reason="torch/transformers (and the model weights) are installed in this "
           "environment -- this test specifically verifies the degrade-cleanly "
           "behaviour when they're NOT installed, which doesn't apply here. "
           "Skipped, not failed: the precondition genuinely doesn't hold.",
)


@_skip_if_torch_installed
def test_codet5_unavailable_without_torch():
    """
    This suite normally runs in an environment with no torch/transformers
    installed, on purpose (see the module docstring). If this assertion
    ever fails (rather than skips) in that environment, it means
    torch/transformers ended up in requirements.txt by accident -- exactly
    the mistake this feature was built to avoid.
    """
    assert codet5_available() is False


@_skip_if_torch_installed
def test_generate_refactor_returns_none_without_torch():
    result = generate_refactor("def f(x):\n    return x\n")
    assert result is None


@_skip_if_torch_installed
def test_health_reports_codet5_available_false():
    client = app.test_client()
    data = client.get("/health").get_json()
    assert "codet5_available" in data
    assert data["codet5_available"] is False


@_skip_if_torch_installed
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
    assert data["codet5_requested"] is True  # the request DID ask for it
    assert data["codet5_suggestion"] is None
    # every other field must still be present and normal -- this feature
    # being off must not have any effect on the rest of the response.
    # code.strip() because /refactor strips the input before echoing it
    # back, same as every other endpoint in this file.
    assert data["original_code"] == code.strip()
    assert "refactored_code" in data


def test_codet5_requested_reflects_the_actual_request():
    """
    codet5_requested exists specifically so the frontend can tell "wasn't
    asked for" apart from "asked for, but produced nothing" -- both give
    codet5_suggestion: null, so without this field they're indistinguishable.
    Confirmed on both sides here, not just the True case above.
    """
    client = app.test_client()
    code = "def f(x):\n    return x + 1\n"
    resp = client.post("/refactor", json={"code": code, "use_ai": False, "use_codet5": False})
    data = resp.get_json()
    assert data["codet5_requested"] is False
    assert "codet5" not in data["summary"].lower()


def test_refactor_syntax_error_path_has_consistent_codet5_shape():
    client = app.test_client()
    resp = client.post("/refactor", json={"code": "def f(:\n", "use_ai": False, "use_codet5": True})
    data = resp.get_json()
    assert data["codet5_suggestion"] is None
    assert "codet5_available" in data
    assert data["codet5_requested"] is True


@pytest.mark.skipif(
    not codet5_available(),
    reason="torch/transformers/the model weights aren't installed in this "
           "environment -- this test needs the real thing working, not the "
           "degrade-cleanly path the rest of this file covers.",
)
def test_refactor_with_codet5_actually_generates_when_available():
    """
    The mirror image of the degrade-cleanly tests above: when the optional
    dependencies genuinely ARE installed (as they are on a machine that ran
    `pip install -r requirements-optional.txt`), the toggle must actually
    do something real, not just avoid crashing. Runs against the real
    fine-tuned weights whenever this environment has them -- the same
    guarantee CODET5_MANUAL_CHECKLIST.md's manual verification checked by
    hand, now automated for any environment where it can be.
    """
    client = app.test_client()
    # A trained-family shape (membership_to_set) -- known to generate a
    # genuinely correct, verified rewrite, unlike a held-out-family input
    # which is expected to fail verification some of the time.
    code = (
        "def find_common(items, pool):\n"
        "    output = []\n"
        "    for item in items:\n"
        "        if item in pool:\n"
        "            output.append(item)\n"
        "    return output\n"
    )
    resp = client.post("/refactor", json={"code": code, "use_ai": False, "use_codet5": True})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["codet5_available"] is True
    assert data["codet5_requested"] is True
    suggestion = data["codet5_suggestion"]
    assert suggestion is not None, "generation produced nothing -- model or generation itself is broken"
    assert suggestion["validated"] is True
    assert suggestion["verified"] is True, suggestion.get("verification_note")
    # verified=True gates the performance comparison (see
    # _build_codet5_suggestion in main.py) -- confirm that gate actually
    # ran, not just that the field exists.
    assert suggestion["performance"] is not None
    assert suggestion["performance"]["measured"] is True
