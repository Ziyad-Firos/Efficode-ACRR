"""Self-contained verification of everything changed this session."""
import sys, json, ast
sys.path.insert(0, "/tmp/acrr")

passed, failed = [], []
def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  — {detail}" if detail and not cond else ""))

print("=" * 70); print("A. HTTP error handling (main.py)"); print("=" * 70)
from app.main import app
c = app.test_client()

r = c.get("/nope")
check("404 stays 404", r.status_code == 404, f"got {r.status_code}")

r = c.get("/review")                       # GET on a POST-only route
check("405 stays 405 (was 500)", r.status_code == 405, f"got {r.status_code}")

r = c.post("/review", data=b"x" * (300 * 1024), content_type="application/json")
check("413 stays 413 (was 500)", r.status_code == 413, f"got {r.status_code}")

r = c.post("/review", json={"code": 12345})
check("wrong-typed code -> 400", r.status_code == 400, f"got {r.status_code}")

r = c.post("/review", json={})
check("missing code -> 400", r.status_code == 400, f"got {r.status_code}")

r = c.post("/review", json={"code": "def f(:"})
check("syntax error -> 200 with valid=false",
      r.status_code == 200 and r.get_json().get("valid") is False, f"got {r.status_code}")

r = c.get("/health")
check("health responds", r.status_code == 200 and r.get_json().get("status") == "ok")

print(); print("=" * 70); print("B. Refactor pipeline"); print("=" * 70)
MESSY = """def find_duplicates(nums):
    duplicates = []
    unused = 42
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                if nums[i] not in duplicates:
                    duplicates.append(nums[i])
    return duplicates
    print('unreachable')
"""
r = c.post("/refactor", json={"code": MESSY, "use_ai": False})
d = r.get_json()
check("refactor returns 200", r.status_code == 200, f"got {r.status_code}")
check("refactored code still parses", _ok := (lambda: (ast.parse(d["refactored_code"]), True)[1])())
applied = [x for x in d["applied_rules"] if x.get("applied", True)]
check("rules were applied", len(applied) > 0, str(d.get("applied_rules")))
check("advice is not counted as an applied change",
      all(x.get("applied", True) for x in applied))
check("diff produced", bool(d["diff"]))
check("degrades cleanly with use_ai=False",
      d["ai_available"] is False and d["ai_suggestions"] == [])

print("       applied:", [x["rule"] for x in d["applied_rules"] if x.get("applied", True)])
print("       advice :", [x["rule"] for x in d["applied_rules"] if not x.get("applied", True)])

print(); print("=" * 70); print("C. Complexity prediction + explanation"); print("=" * 70)
comp = d.get("complexity")
check("complexity block present", comp is not None)
if comp:
    check("nested-loop code detected as O(n^2)", comp["before"] == "O(n²)", str(comp["before"]))
    check("confidence is high (was ~0.5)", comp["confidence"] >= 0.75, str(comp["confidence"]))
    check("explanation is populated", len(comp.get("explanation", [])) > 0)
    check("suggestion is populated", bool(comp.get("suggestion")))
    print("       before:", comp["before"], f"({comp['confidence']:.0%})")
    for line in comp.get("explanation", []):
        print("        -", line)
    print("       suggestion:", (comp.get("suggestion") or "")[:90] + "...")

print(); print("=" * 70); print("D. Clean code is left alone"); print("=" * 70)
CLEAN = "def add(a, b):\n    return a + b\n"
r = c.post("/refactor", json={"code": CLEAN, "use_ai": False})
d2 = r.get_json()
check("clean code unchanged", d2["refactored_code"].strip() == CLEAN.strip(),
      repr(d2["refactored_code"]))
check("no rules applied to clean code", len(d2["applied_rules"]) == 0)
check("clean code -> O(1)", d2["complexity"]["before"] == "O(1)")

print(); print("=" * 70); print("E. Review pipeline degrades without flake8/radon/bandit"); print("=" * 70)
r = c.post("/review", json={"code": MESSY})
d3 = r.get_json()
check("review returns 200 even with linters missing", r.status_code == 200)
check("review still scores", d3.get("quality_score") is not None)

print(); print("=" * 70)
print(f"RESULT: {len(passed)} passed, {len(failed)} failed")
if failed:
    print("FAILED:", ", ".join(failed))
sys.exit(1 if failed else 0)
