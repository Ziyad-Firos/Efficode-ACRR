"""
review/style.py — Style and lint checks via flake8.

Runs flake8 as a subprocess against a temp file.
Returns structured Issue objects, never raw text.

Timeout: 10 seconds. If flake8 hangs or isn't installed,
returns an empty list with a warning — never crashes the pipeline.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import os
import re
from typing import List

from app.models import Issue, IssueCategory, IssueSeverity

logger = logging.getLogger("acrr.review.style")

# flake8 codes → severity mapping
_ERROR_CODES = {"E", "F"}     # syntax errors, undefined names
_WARNING_CODES = {"W", "C"}   # warnings, conventions
_INFO_CODES = {"B", "N"}      # bugbear, naming


def _severity_for_code(code: str) -> IssueSeverity:
    prefix = code[0].upper() if code else "W"
    if prefix in _ERROR_CODES:
        return IssueSeverity.ERROR
    if prefix in _WARNING_CODES:
        return IssueSeverity.WARNING
    return IssueSeverity.INFO


def analyze_style(code: str) -> List[Issue]:
    """
    Run flake8 on the given code and return structured issues.

    Args:
        code: Python source string (already syntax-validated).

    Returns:
        List of Issue objects. Empty list on tool failure.
    """
    issues: List[Issue] = []

    # Write code to a temp file — flake8 needs a real file path
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
        logger.warning("Could not create temp file for style check: %s", exc)
        return issues

    try:
        result = subprocess.run(
            [
                "flake8",
                "--max-line-length=120",
                "--format=%(row)d:%(col)d:%(code)s:%(text)s",
                "--extend-ignore=E501",   # ignore long lines (handled by max-line-length above)
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        for line in result.stdout.splitlines():
            issue = _parse_flake8_line(line)
            if issue:
                issues.append(issue)

        logger.debug("style check: %d issues found", len(issues))

    except FileNotFoundError:
        logger.warning("flake8 not found — style check skipped. Install with: pip install flake8")
    except subprocess.TimeoutExpired:
        logger.warning("flake8 timed out — style check skipped")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error in style check: %s", exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return issues


def _parse_flake8_line(line: str) -> Issue | None:
    """
    Parse one line of flake8 output in format:
        ROW:COL:CODE:MESSAGE
    e.g.  3:5:W291:trailing whitespace
    """
    # Format: row:col:code:text
    pattern = re.compile(r"^(\d+):(\d+):([A-Z]\d+):(.+)$")
    match = pattern.match(line.strip())
    if not match:
        return None

    row, col, code, message = match.groups()
    return Issue(
        line=int(row),
        column=int(col),
        severity=_severity_for_code(code),
        category=IssueCategory.STYLE,
        rule=code,
        message=message.strip(),
    )
