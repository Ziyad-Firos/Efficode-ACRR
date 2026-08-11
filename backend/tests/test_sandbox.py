"""
tests/test_sandbox.py — cross-platform isolated execution correctness.

Why this file exists
--------------------
app/verify/sandbox.py is the one place in this project that executes
submitted code for real, which makes it the one module where "looks right"
and "is actually safe" can diverge badly. Building it surfaced a real bug on
this machine: this venv's python.exe launches the real interpreter as a
CHILD process rather than exec'ing in place, so a naive memory-usage check
against only the immediate subprocess PID silently watched the wrong
process (RSS stayed flat at ~4MB while the actual worker consumed real
memory in a child of that PID). The fix sums RSS across the whole process
tree. This file exists so that regression can never come back silently.

Every test here that spawns a process is slow-ish (subprocess startup +
Python interpreter startup, twice per test on this environment because of
the child-of-child behaviour above) -- expect this file to take longer
than the rest of the suite. That's the cost of testing the real thing
instead of mocking subprocess, which would not have caught the bug above.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.verify.sandbox import run_isolated  # noqa: E402


def test_normal_execution_returns_correct_value():
    result = run_isolated("def add(a, b):\n    return a + b\n", "add(2, 3)", timeout=5)
    assert result.ok is True
    assert result.return_value == 5


def test_exception_inside_code_is_captured_not_raised():
    """A ZeroDivisionError in the sandboxed code must come back as a
    result field, never propagate out and crash the caller."""
    result = run_isolated("def divide(a, b):\n    return a / b\n", "divide(1, 0)", timeout=5)
    assert result.ok is False
    assert result.timed_out is False
    assert "ZeroDivisionError" in result.error


def test_infinite_loop_is_killed_by_timeout():
    """The specific guarantee this module exists for: a hang gets killed,
    reliably, regardless of what the code inside it is doing. Uses a short
    timeout so the test itself doesn't hang the suite."""
    t0 = time.time()
    result = run_isolated(
        "def loop_forever():\n    while True:\n        pass\n",
        "loop_forever()",
        timeout=2,
    )
    elapsed = time.time() - t0
    assert result.ok is False
    assert result.timed_out is True
    assert elapsed < 4, f"took {elapsed:.1f}s to kill a 2s-timeout process -- kill isn't working"


def test_memory_bomb_is_killed_by_memory_limit_not_timeout():
    """Regression test for the process-tree bug: must be caught quickly by
    the memory poller, not fall through to the (much slower) timeout."""
    t0 = time.time()
    result = run_isolated(
        "def bomb():\n    x = []\n    while True:\n        x.append(bytearray(10**7))\n",
        "bomb()",
        timeout=10,
        memory_limit_mb=100,
    )
    elapsed = time.time() - t0
    assert result.ok is False
    assert result.memory_exceeded is True
    assert result.timed_out is False
    assert elapsed < 5, (
        f"took {elapsed:.1f}s -- memory bomb was caught by the timeout fallback, "
        "not the memory poller (the process-tree RSS bug is back)"
    )


def test_recursive_but_fast_code_completes_normally():
    """A CPU-bound function that finishes well within the timeout must not
    be mistaken for a hang."""
    result = run_isolated(
        "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
        "fib(20)",
        timeout=5,
    )
    assert result.ok is True
    assert result.return_value == 6765


def test_network_access_is_blocked():
    result = run_isolated(
        "import socket\ndef connect():\n    return socket.socket()\n",
        "connect()",
        timeout=5,
    )
    assert result.ok is False
    assert "network access is disabled" in result.error


def test_file_write_is_blocked():
    result = run_isolated(
        "def write_file():\n"
        "    with open('evil.txt', 'w') as f:\n"
        "        f.write('x')\n"
        "    return 'wrote'\n",
        "write_file()",
        timeout=5,
    )
    assert result.ok is False
    assert "file writes are disabled" in result.error


def test_file_read_is_still_allowed():
    """Only write modes are blocked -- reading must still work, since
    differential testing needs ordinary code (open a file, read it) to
    behave normally when that isn't the thing under test."""
    result = run_isolated(
        "def read_self():\n"
        "    with open(__file__, 'r') as f:\n"
        "        return len(f.read()) > 0\n",
        "read_self()",
        timeout=5,
    )
    assert result.ok is True
    assert result.return_value is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
