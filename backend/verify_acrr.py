"""
ACRR Verification Suite
=======================
Hits the live server and checks each component does correct,
non-stub work — not just that it returns 200.

Run:
    .venv\\Scripts\\python.exe verify_acrr.py
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import requests

BASE = os.environ.get("ACRR_BASE_URL", "http://localhost:8000")
results = []


def record(name, passed, detail=""):
    results.append((name, passed, detail))
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")


def post(path, payload, timeout=15):
    return requests.post(f"{BASE}{path}", json=payload, timeout=timeout)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── 1. Health ─────────────────────────────────────────────────────────────

def test_health():
    section("1. HEALTH CHECK")
    r = requests.get(f"{BASE}/health", timeout=5)
    body = r.json()
    record("Server responds to /health", r.status_code == 200, f"status={r.status_code}")
    record("AI configured = True (key in .env)",
           body.get("ai_configured") is True,
           f"ai_configured={body.get('ai_configured')}")
    record("Python version field present", "version" in body, str(body))


# ── 2. Review engine ──────────────────────────────────────────────────────

def test_style_issues():
    section("2a. REVIEW — Style (flake8)")
    code = (
        "import os\nimport sys\n\n"
        "def add(a,b):\n"
        "    unused = 42\n"
        "    return a+b\n"
    )
    r = post("/review", {"code": code})
    body = r.text
    record("Unused import os caught (F401)",
           r.status_code == 200 and "F401" in body,
           f"F401 in response: {'F401' in body}")
    record("Unused local variable caught (F841)",
           "F841" in body,
           f"F841 in response: {'F841' in body}")
    record("Valid response shape (valid/issues/quality_score)",
           all(k in body for k in ["valid", "issues", "quality_score"]),
           body[:200])


def test_security_issues():
    section("2b. REVIEW — Security (bandit)")
    code = (
        "def run(user_input):\n"
        "    password = 'hunter2'\n"
        "    eval(user_input)\n"
        "    return password\n"
    )
    r = post("/review", {"code": code})
    body = r.text.lower()
    record("eval() usage flagged by bandit",
           r.status_code == 200 and "eval" in body,
           f"'eval' in response: {'eval' in body}")
    record("Hardcoded password flagged",
           "password" in body or "hardcoded" in body or "b105" in body,
           f"body preview: {r.text[:300]}")


def test_smells():
    section("2c. REVIEW — Code smells (AST)")
    code = (
        "def bad(list, data=[]):\n"
        "    for i in range(10):\n"
        "        for j in range(10):\n"
        "            for k in range(10):\n"
        "                pass\n"
        "    try:\n"
        "        pass\n"
        "    except:\n"
        "        pass\n"
        "    return data\n"
    )
    r = post("/review", {"code": code})
    body = r.text
    record("Deep nesting flagged (SMELL002)",
           "SMELL002" in body,
           f"SMELL002 in response: {'SMELL002' in body}")
    record("Bare except flagged (SMELL008)",
           "SMELL008" in body,
           f"SMELL008 in response: {'SMELL008' in body}")
    record("Mutable default argument flagged (SMELL009)",
           "SMELL009" in body,
           f"SMELL009 in response: {'SMELL009' in body}")
    record("Builtin shadow flagged (SMELL010)",
           "SMELL010" in body,
           f"SMELL010 in response: {'SMELL010' in body}")


def test_complexity():
    section("2d. REVIEW — Complexity (radon)")
    # Trivial: CC=1. Deeply branchy: CC should be 5+
    trivial = "def f(x):\n    return x + 1\n"
    branchy = (
        "def classify(x):\n"
        "    if x > 100:\n        return 'huge'\n"
        "    elif x > 50:\n        return 'big'\n"
        "    elif x > 20:\n        return 'medium'\n"
        "    elif x > 10:\n        return 'small'\n"
        "    elif x > 0:\n        return 'tiny'\n"
        "    elif x == 0:\n        return 'zero'\n"
        "    else:\n        return 'negative'\n"
    )
    r1 = post("/review", {"code": trivial})
    r2 = post("/review", {"code": branchy})
    s1 = r1.json().get("quality_score", {}).get("breakdown", {}).get("complexity", -1)
    s2 = r2.json().get("quality_score", {}).get("breakdown", {}).get("complexity", -1)
    record("Trivial function gets high complexity score (>= 90)",
           s1 >= 90,
           f"trivial complexity score = {s1}")
    record("Branchy function scores lower than trivial",
           s2 < s1,
           f"trivial={s1}  branchy={s2}")
    record("Responses differ (not constant/stubbed output)",
           r1.text != r2.text,
           f"responses identical: {r1.text == r2.text}")


def test_quality_score():
    section("2e. REVIEW — Quality score assembly")
    clean = "def add(a, b):\n    return a + b\n"
    messy = (
        "import os, sys\n"
        "PASSWORD = 'abc123'\n"
        "def f(list, data=[]):\n"
        "    for i in range(10):\n"
        "        for j in range(10):\n"
        "            eval('x')\n"
        "    unused = 42\n"
        "    return data\n"
    )
    r1 = post("/review", {"code": clean})
    r2 = post("/review", {"code": messy})
    s_clean = r1.json().get("quality_score", {}).get("score", -1)
    s_messy = r2.json().get("quality_score", {}).get("score", -1)
    g_clean = r1.json().get("quality_score", {}).get("grade", "?")
    g_messy = r2.json().get("quality_score", {}).get("grade", "?")
    record("Clean code scores higher than messy code",
           s_clean > s_messy,
           f"clean={s_clean}({g_clean})  messy={s_messy}({g_messy})")
    record("Clean code gets grade A or B",
           g_clean in ("A", "B"),
           f"clean grade = {g_clean}")
    record("Messy code gets grade C or worse",
           g_messy in ("C", "D", "F"),
           f"messy grade = {g_messy}")


# ── 3. Rule-based refactor ────────────────────────────────────────────────

def test_constant_folding():
    section("3a. REFACTOR — Constant folding")
    code = "def compute():\n    x = 2 + 3 * 4\n    y = 2 ** 10\n    return x + y\n"
    r = post("/refactor", {"code": code, "use_ai": False})
    out = r.json().get("refactored_code", "")
    rules = [rl["rule"] for rl in r.json().get("applied_rules", [])]
    record("2 + 3*4 folded to 14",
           "14" in out and "2 + 3 * 4" not in out,
           f"refactored:\n{out}")
    record("2**10 folded to 1024",
           "1024" in out and "2 ** 10" not in out,
           f"refactored:\n{out}")
    record("constant_fold in applied_rules",
           "constant_fold" in rules,
           f"applied_rules: {rules}")


def test_dead_code_removal():
    section("3b. REFACTOR — Dead code removal")
    code = (
        "def f(flag):\n"
        "    if flag:\n"
        "        return 'yes'\n"
        "    print('reachable')\n"
        "    return 'no'\n"
        "    print('unreachable')\n"
        "    x = 999\n"
    )
    r = post("/refactor", {"code": code, "use_ai": False})
    out = r.json().get("refactored_code", "")
    record("Unreachable print() removed",
           "unreachable" not in out,
           f"refactored:\n{out}")
    record("x = 999 removed",
           "x = 999" not in out,
           f"refactored:\n{out}")
    record("Reachable print() kept",
           "reachable" in out,
           f"refactored:\n{out}")
    record("Return value kept",
           "return 'yes'" in out or "return" in out,
           f"refactored:\n{out}")


def test_unused_variable_removal():
    section("3c. REFACTOR — Unused variable removal")
    code = "def calc(a, b):\n    temp = a * 2\n    result = a + b\n    return result\n"
    r = post("/refactor", {"code": code, "use_ai": False})
    out = r.json().get("refactored_code", "")
    record("Unused variable 'temp' removed",
           "temp" not in out,
           f"refactored:\n{out}")
    record("Used variable 'result' kept",
           "result" in out,
           f"refactored:\n{out}")


def test_condition_simplification():
    section("3d. REFACTOR — Condition simplification")
    code = (
        "def check(x):\n"
        "    if x == True:\n"
        "        return 1\n"
        "    if x == False:\n"
        "        return 0\n"
        "    return -1\n"
    )
    r = post("/refactor", {"code": code, "use_ai": False})
    out = r.json().get("refactored_code", "")
    record("'x == True' simplified away",
           "== True" not in out,
           f"refactored:\n{out}")
    record("'x == False' simplified away",
           "== False" not in out,
           f"refactored:\n{out}")


def test_clean_code_untouched():
    section("3e. REFACTOR — Clean code not mangled (control case)")
    code = "def add(a, b):\n    return a + b\n"
    r = post("/refactor", {"code": code, "use_ai": False})
    out = r.json().get("refactored_code", "")
    applied = r.json().get("applied_rules", [])
    record("'def add' still present in output",
           "def add" in out,
           f"refactored:\n{out}")
    record("'return a + b' still present",
           "a + b" in out or "a+b" in out,
           f"refactored:\n{out}")
    record("No spurious rules applied to clean code",
           len(applied) == 0,
           f"applied_rules: {applied}")


def test_diff_is_populated():
    section("3f. REFACTOR — Diff is non-empty when changes made")
    code = "def f():\n    x = 2 * 8\n    unused = 99\n    return 0\n"
    r = post("/refactor", {"code": code, "use_ai": False})
    body = r.json()
    diff = body.get("diff", "")
    record("Diff is non-empty when code changed",
           bool(diff.strip()),
           f"diff (first 200 chars):\n{diff[:200]}")
    record("Diff contains --- and +++ markers",
           "---" in diff and "+++" in diff,
           f"markers present: {'---' in diff} / {'+++' in diff}")


# ── 4. ML complexity predictor ────────────────────────────────────────────

def test_ml_predictions_vary():
    section("4. ML — Complexity predictor varies with input")
    samples = {
        "O(1) constant": "def get_first(lst):\n    return lst[0]\n",
        "O(n) single loop": (
            "def find_max(lst):\n"
            "    m = lst[0]\n"
            "    for x in lst:\n"
            "        if x > m:\n"
            "            m = x\n"
            "    return m\n"
        ),
        "O(n^2) nested loops": (
            "def bubble_sort(arr):\n"
            "    n = len(arr)\n"
            "    for i in range(n):\n"
            "        for j in range(n - i - 1):\n"
            "            if arr[j] > arr[j+1]:\n"
            "                arr[j], arr[j+1] = arr[j+1], arr[j]\n"
            "    return arr\n"
        ),
        "O(n^3) triple loop": (
            "def triple(arr):\n"
            "    for i in arr:\n"
            "        for j in arr:\n"
            "            for k in arr:\n"
            "                pass\n"
        ),
    }
    predictions = {}
    for label, code in samples.items():
        r = post("/refactor", {"code": code, "use_ai": False})
        body = r.json()
        cx = body.get("complexity")
        pred = cx.get("before") if cx else "null (sklearn not installed)"
        predictions[label] = pred
        print(f"         {label:30s} -> {pred}")

    distinct = len(set(predictions.values()))
    record("ML predictions vary across complexity classes (not all same)",
           distinct > 1,
           f"distinct predictions: {distinct} out of {len(samples)} inputs\n"
           f"predictions: {predictions}")
    record("O(1) not predicted as O(n²) or worse",
           predictions.get("O(1) constant") not in ("O(n²)", "O(n³+)"),
           f"O(1) prediction: {predictions.get('O(1) constant')}")
    record("O(n^2) not predicted as O(1)",
           predictions.get("O(n^2) nested loops") != "O(1)",
           f"O(n²) prediction: {predictions.get('O(n^2) nested loops')}")


# ── 5. AI layer ───────────────────────────────────────────────────────────

def test_ai_behavior():
    section("5. AI LAYER — Fallback and response structure")
    code = (
        "def two_sum(nums, target):\n"
        "    for i in range(len(nums)):\n"
        "        for j in range(i + 1, len(nums)):\n"
        "            if nums[i] + nums[j] == target:\n"
        "                return [i, j]\n"
    )
    r = post("/refactor", {"code": code, "use_ai": True}, timeout=25)
    record("Request completes (no 500 crash)",
           r.status_code == 200,
           f"status={r.status_code}")
    if r.status_code != 200:
        return
    body = r.json()
    ai_available = body.get("ai_available")
    record("rule-based results present regardless of AI state",
           bool(body.get("refactored_code")),
           f"ai_available={ai_available}")
    record("ai_available field is a bool (not missing/null)",
           isinstance(ai_available, bool),
           f"ai_available={ai_available!r}")
    record("ai_suggestions is a list (not null/missing)",
           isinstance(body.get("ai_suggestions"), list),
           f"ai_suggestions type: {type(body.get('ai_suggestions')).__name__}")
    if ai_available is False:
        record("ai_suggestions is empty list when unavailable (quota or no key)",
               body.get("ai_suggestions") == [],
               "EXPECTED TODAY — quota resets at midnight Pacific. Run again after reset.")
    else:
        suggs = body.get("ai_suggestions", [])
        record("At least one AI suggestion returned",
               len(suggs) > 0,
               f"suggestions count: {len(suggs)}")
        if suggs:
            record("First suggestion has explanation and diff fields",
                   "explanation" in suggs[0] and "diff" in suggs[0],
                   f"keys: {list(suggs[0].keys())}")
            record("First suggestion is validated (valid Python)",
                   suggs[0].get("validated") is True,
                   f"validated={suggs[0].get('validated')}")
            print(f"         AI explanation: {suggs[0].get('explanation', '')[:200]}")


# ── 6. Reliability / adversarial ──────────────────────────────────────────

def test_reliability():
    section("6. RELIABILITY — Adversarial inputs")

    # Empty code
    r = post("/review", {"code": ""})
    record("Empty code: no 500",
           r.status_code != 500,
           f"status={r.status_code}")

    # Syntax error
    r = post("/review", {"code": "def broken(:\n    pass\n"})
    record("Broken syntax: no 500",
           r.status_code != 500,
           f"status={r.status_code}")
    record("Broken syntax: valid=false in response",
           r.status_code == 200 and r.json().get("valid") is False,
           f"valid={r.json().get('valid') if r.status_code == 200 else 'N/A'}")

    # Wrong type
    r = post("/review", {"code": 12345})
    record("Wrong type for 'code': rejected (400/422, not 500)",
           r.status_code in (400, 422),
           f"status={r.status_code}")

    # Missing field
    r = post("/review", {})
    record("Missing 'code' field: rejected (400/422)",
           r.status_code in (400, 422),
           f"status={r.status_code}")

    # Unicode
    r = post("/review", {"code": "# caf\u00e9 \u2615\ndef f():\n    return '\u65e5\u672c\u8a9e'\n"})
    record("Unicode in code: no 500",
           r.status_code != 500,
           f"status={r.status_code}")

    # Large input
    big = "\n".join(f"x_{i} = {i}" for i in range(3000))
    start = time.time()
    try:
        r = post("/review", {"code": big}, timeout=20)
        elapsed = time.time() - start
        record("3000-line input handled within 20s",
               r.status_code != 500,
               f"status={r.status_code}, time={elapsed:.1f}s")
    except requests.exceptions.Timeout:
        record("3000-line input handled within 20s",
               False,
               "timed out — no size/timeout guard on the server")


def test_concurrent_requests():
    section("7. RELIABILITY — Concurrent requests don't cross-contaminate")
    # Three different inputs each with a unique unused import name
    samples = [
        ("import moduleAAA\ndef f():\n    pass\n", "moduleAAA", "moduleBBB", "moduleCCC"),
        ("import moduleBBB\ndef g():\n    pass\n", "moduleBBB", "moduleAAA", "moduleCCC"),
        ("import moduleCCC\ndef h():\n    pass\n", "moduleCCC", "moduleAAA", "moduleBBB"),
    ]

    def run(args):
        code, own, *others = args
        r = post("/review", {"code": code})
        return r.text, own, others

    with ThreadPoolExecutor(max_workers=3) as pool:
        results_raw = list(pool.map(run, samples))

    ok = True
    for text, own, others in results_raw:
        if own not in text:
            ok = False
        for other in others:
            if other in text:
                ok = False

    record("Concurrent requests don't bleed into each other",
           ok,
           "failure = shared temp-file names across subprocess calls")


# ── Runner ────────────────────────────────────────────────────────────────

def main():
    print(f"ACRR Verification Suite  |  target: {BASE}")

    suites = [
        test_health,
        test_style_issues,
        test_security_issues,
        test_smells,
        test_complexity,
        test_quality_score,
        test_constant_folding,
        test_dead_code_removal,
        test_unused_variable_removal,
        test_condition_simplification,
        test_clean_code_untouched,
        test_diff_is_populated,
        test_ml_predictions_vary,
        test_ai_behavior,
        test_reliability,
        test_concurrent_requests,
    ]

    for suite in suites:
        try:
            suite()
        except Exception as e:
            record(suite.__name__, False, f"suite raised {type(e).__name__}: {e}")

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  RESULT: {passed}/{total} checks passed")
    if passed < total:
        print("\n  FAILURES:")
        for name, p, detail in results:
            if not p:
                print(f"    FAIL: {name}")
                if detail:
                    print(f"          {detail.splitlines()[0]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
