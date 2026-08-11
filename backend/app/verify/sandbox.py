"""
verify/sandbox.py — cross-platform isolated code execution.

Why this exists
----------------
Every other feature in this project only READS submitted code (parses it,
extracts features, asks an AI to comment on it). Differential testing of
refactorings needs to RUN it — there is no way to prove two versions of a
function behave the same without executing both and comparing outputs.
This is a deliberate, documented exception to this project's principle that
nothing submitted is ever executed. It exists ONLY inside this sandbox, and
ONLY for the verification pipeline — no other endpoint executes anything.

The one thing this module exists to guarantee, on ANY operating system:
code that hangs (infinite loop, blocking I/O) or tries to exhaust memory
gets killed, reliably, without taking the rest of the process down with it.

Design choices, and why they're not what a first draft would reach for
------------------------------------------------------------------------
- subprocess, not exec()/threading. An infinite `while True: pass` in the
  SAME process cannot be reliably interrupted from another thread in
  CPython — a tight loop can starve everything else indefinitely. Killing
  a separate OS PROCESS is always reliable, regardless of what the code
  inside it is doing. That is the actual reason for subprocess isolation
  here, not a vague "for safety" gesture.

- subprocess timeout, not resource.setrlimit, for the time limit.
  resource.RLIMIT_CPU is POSIX-only — it does not exist on Windows at all.
  Popen + a manually-enforced timeout (see run_isolated) kills the child
  process on every platform Python supports. This is the PRIMARY,
  always-reliable defense against a runaway/infinite loop, on any OS —
  the actual requirement this module was built to satisfy.

- psutil for the memory limit, not resource.RLIMIT_AS. Same problem:
  RLIMIT_AS is POSIX-only. psutil is a genuinely cross-platform library
  (Windows/Linux/macOS) that can read a running process's actual memory
  usage, so a background thread polls the child's RSS and kills it the
  moment it crosses the limit. This is reactive (polling), not a hard
  OS-enforced ceiling like RLIMIT_AS — said honestly, not oversold — but
  it is the same guarantee on every OS, which a POSIX-only hard limit
  cannot offer.

- Where POSIX resource limits ARE available (Linux — wherever this
  deploys, Render included) they're applied too, inside the child, as a
  second, stronger layer underneath the cross-platform one above. Never
  relied on as the only layer, since it silently does nothing on Windows.

What this does NOT claim to be
-------------------------------
A hardened security sandbox (no seccomp, no containers, no gVisor). It
blocks network sockets and file writes at the Python level (monkeypatching
before the submitted code runs), which stops accidental/casual misuse and
is honest about not stopping a determined attacker working at the C/ctypes
level. The realistic threat model here is "someone pastes an infinite loop
or a memory bomb by accident, or a generated refactoring has a bug" — not
nation-state sandbox escape. Stated plainly so nobody mistakes this
module for more than it is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MEMORY_LIMIT_MB = 256
_MEMORY_POLL_INTERVAL_SECONDS = 0.05

_RESULT_START = "__SANDBOX_RESULT_START__"
_RESULT_END = "__SANDBOX_RESULT_END__"


@dataclass
class SandboxResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    timed_out: bool = False
    memory_exceeded: bool = False
    exit_code: Optional[int] = None
    error: Optional[str] = None


_RUNNER_PREAMBLE = """\
# --- sandbox preamble: runs before the submitted code, in the child process ---
import builtins as _builtins
import json as _json
import socket as _socket

# Best-effort restriction against accidental/casual network and file-write
# misuse. NOT a hard security boundary (see this module's docstring) --
# stops "the pasted code happens to call requests.get()", not a
# deliberately obfuscated bypass.
def _blocked_socket(*_a, **_kw):
    raise OSError("network access is disabled in the verification sandbox")
_socket.socket = _blocked_socket

_orig_open = _builtins.open
def _guarded_open(file, mode='r', *a, **kw):
    if any(m in mode for m in ('w', 'a', 'x', '+')):
        raise OSError("file writes are disabled in the verification sandbox")
    return _orig_open(file, mode, *a, **kw)
_builtins.open = _guarded_open

# POSIX-only extra layer -- silently absent on Windows. The subprocess
# timeout (parent process) and psutil poller (parent process) are what
# this module actually relies on for the time and memory limits on every
# OS; this is a stronger layer underneath them where the platform allows it.
try:
    import resource as _resource
    _resource.setrlimit(_resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))
    _resource.setrlimit(_resource.RLIMIT_AS, ({memory_bytes}, {memory_bytes}))
except Exception:
    pass
# --- end preamble ---

"""


def _build_runner_script(code: str, call_expr: str, cpu_seconds: int, memory_bytes: int) -> str:
    preamble = _RUNNER_PREAMBLE.format(cpu_seconds=cpu_seconds, memory_bytes=memory_bytes)
    footer = textwrap.dedent(f"""
        if __name__ == "__main__":
            try:
                _result = {call_expr}
                print({_RESULT_START!r})
                print(_json.dumps({{"ok": True, "value": _result}}, default=repr))
                print({_RESULT_END!r})
            except Exception as _exc:
                print({_RESULT_START!r})
                print(_json.dumps({{"ok": False, "error": f"{{type(_exc).__name__}}: {{_exc}}"}}))
                print({_RESULT_END!r})
        """)
    return preamble + "\n" + code + "\n" + footer


def _extract_result(stdout: str) -> Optional[dict]:
    if _RESULT_START not in stdout or _RESULT_END not in stdout:
        return None
    start = stdout.index(_RESULT_START) + len(_RESULT_START)
    end = stdout.index(_RESULT_END)
    payload = stdout[start:end].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def run_isolated(
    code: str,
    call_expr: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> SandboxResult:
    """
    Execute `code` (a full module's worth of Python source, e.g. a function
    definition) followed by evaluating `call_expr` (e.g.
    "sum_list([1, 2, 3])") in an isolated subprocess, and return its result.

    Guarantees, on every OS Python supports:
      - the process is killed if it runs longer than `timeout` seconds
        (catches infinite loops and hangs) -- via a manually-enforced
        Popen timeout, not resource.setrlimit.
      - the process is killed if its memory usage exceeds `memory_limit_mb`
        (catches memory bombs) -- via a psutil polling thread, not
        resource.setrlimit.
    Neither guarantee depends on a platform-specific API being available.
    """
    with tempfile.TemporaryDirectory(prefix="acrr_sandbox_") as tmpdir:
        runner_path = Path(tmpdir) / "runner.py"
        script = _build_runner_script(
            code, call_expr,
            cpu_seconds=int(timeout) + 1,
            memory_bytes=memory_limit_mb * 1024 * 1024,
        )
        runner_path.write_text(script, encoding="utf-8")

        # Minimal environment -- no inherited secrets, no PYTHONPATH tricks.
        env = {"PATH": os.environ.get("PATH", "")}
        if sys.platform == "win32":
            # Windows subprocess creation needs SystemRoot to resolve DLLs.
            env["SystemRoot"] = os.environ.get("SystemRoot", "")

        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", str(runner_path)],
            cwd=tmpdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        memory_exceeded = threading.Event()
        stop_polling = threading.Event()

        def _tree_rss_mb(ps_proc) -> float:
            """
            Sum RSS across ps_proc AND all its descendants, not just
            ps_proc itself. On at least one real Windows setup this
            venv's python.exe launches the actual interpreter as a CHILD
            process rather than exec'ing in place -- confirmed empirically
            (a second python.exe visible in psutil's children(), owning
            100% of the real memory/CPU usage while the immediate child
            stayed flat at ~4MB). Checking only proc.pid directly missed
            all real memory growth on that setup. This is defensive on
            platforms where it isn't needed and load-bearing on the one
            where it is.
            """
            try:
                total = ps_proc.memory_info().rss
            except Exception:
                return 0.0
            try:
                for child in ps_proc.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except Exception:
                        pass  # child exited between listing and reading -- skip it
            except Exception:
                pass
            return total / (1024 * 1024)

        def _poll_memory() -> None:
            try:
                import psutil  # type: ignore
                ps_proc = psutil.Process(proc.pid)
            except Exception:
                return  # psutil unavailable or process already gone -- timeout is still the backstop
            while not stop_polling.is_set():
                if not ps_proc.is_running():
                    return
                rss_mb = _tree_rss_mb(ps_proc)
                if rss_mb > memory_limit_mb:
                    memory_exceeded.set()
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return
                stop_polling.wait(_MEMORY_POLL_INTERVAL_SECONDS)

        poller = threading.Thread(target=_poll_memory, daemon=True)
        poller.start()

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            timed_out = True
        finally:
            stop_polling.set()
            poller.join(timeout=1)

        exit_code = proc.returncode

        if memory_exceeded.is_set():
            return SandboxResult(
                ok=False, stdout=stdout, stderr=stderr,
                memory_exceeded=True, exit_code=exit_code,
                error=f"memory usage exceeded {memory_limit_mb}MB",
            )
        if timed_out:
            return SandboxResult(
                ok=False, stdout=stdout, stderr=stderr,
                timed_out=True, exit_code=exit_code,
                error=f"execution exceeded {timeout}s timeout",
            )

        result = _extract_result(stdout)
        if result is None:
            return SandboxResult(
                ok=False, stdout=stdout, stderr=stderr, exit_code=exit_code,
                error="sandbox produced no result (crashed before completing)",
            )
        if not result.get("ok"):
            return SandboxResult(
                ok=False, stdout=stdout, stderr=stderr, exit_code=exit_code,
                error=result.get("error", "unknown error"),
            )
        return SandboxResult(
            ok=True, stdout=stdout, stderr=stderr,
            return_value=result.get("value"), exit_code=exit_code,
        )
