"""
review/security.py — Security vulnerability detection via bandit.

Bandit is purpose-built for Python security scanning. It catches:
  - eval() / exec() usage
  - Hardcoded passwords / secrets
  - SQL built via string concatenation
  - Use of insecure hash functions (MD5, SHA1)
  - Subprocess shell injection risks
  - assert used for security checks
  - Use of pickle / yaml.load without Loader
  ...and ~40 more patterns.

Runs bandit as a subprocess against a temp file with JSON output,
so parsing is reliable (not regex over human-readable text).
Timeout: 15 seconds.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import List

from app.models import Issue, IssueCategory, IssueSeverity

logger = logging.getLogger("acrr.review.security")

# Bandit severity/confidence → our severity mapping
_BANDIT_SEVERITY: dict[str, IssueSeverity] = {
    "HIGH":   IssueSeverity.ERROR,
    "MEDIUM": IssueSeverity.WARNING,
    "LOW":    IssueSeverity.INFO,
    "UNDEFINED": IssueSeverity.INFO,
}


def analyze_security(code: str) -> List[Issue]:
    """
    Run bandit on the given code and return structured security issues.

    Args:
        code: Python source string (already syntax-validated).

    Returns:
        List of Issue objects. Empty list on tool failure.
    """
    issues: List[Issue] = []

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
    except OSError as exc:
        logger.warning("Could not create temp file for security check: %s", exc)
        return issues

    try:
        result = subprocess.run(
            [
                "bandit",
                "-f", "json",       # machine-readable output
                "-q",               # suppress progress bars
                "--exit-zero",      # don't use exit code for findings (we parse JSON)
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        # bandit exits 1 when it finds issues — that's fine, parse stdout anyway
        if not result.stdout.strip():
            logger.debug("bandit returned no output")
            return issues

        data = json.loads(result.stdout)
        results = data.get("results", [])

        for r in results:
            severity_str = r.get("issue_severity", "LOW").upper()
            severity = _BANDIT_SEVERITY.get(severity_str, IssueSeverity.INFO)

            issues.append(Issue(
                line=r.get("line_number"),
                column=None,
                severity=severity,
                category=IssueCategory.SECURITY,
                rule=r.get("test_id", "B000"),
                message=(
                    f"{r.get('issue_text', 'Security issue')} "
                    f"[confidence: {r.get('issue_confidence', '?')}]"
                ),
            ))

        logger.debug("security check: %d issues found", len(issues))

    except FileNotFoundError:
        logger.warning("bandit not found — security check skipped. Install with: pip install bandit")
    except subprocess.TimeoutExpired:
        logger.warning("bandit timed out — security check skipped")
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse bandit JSON output: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error in security check: %s", exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return issues
