import json, sys, importlib, asyncio, requests, os
from pathlib import Path

BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)

r = {}

# Python
r['python'] = sys.version.split()[0]

# Packages
for pkg in ['flask','flask_cors','flake8','radon','bandit','sklearn','joblib','numpy','pydantic']:
    try:
        m = importlib.import_module(pkg)
        r[f'pkg_{pkg}'] = getattr(m, '__version__', 'ok')
    except: r[f'pkg_{pkg}'] = 'MISSING'
try:
    from google import genai; r['pkg_google_genai'] = 'ok'
except: r['pkg_google_genai'] = 'MISSING'

# .env
key = ''
env_path = BASE_DIR / '.env'
if env_path.exists():
    for line in env_path.read_text(encoding='utf-8').splitlines():
        if line.startswith('GEMINI_API_KEY='): key = line.split('=',1)[1].strip()
r['gemini_key_set'] = bool(key)
r['gemini_key_prefix'] = key[:8] if key else ''

# Model
mp = BASE_DIR / 'app/ml/models/complexity_rf.pkl'
r['model_exists'] = mp.exists()
r['model_kb'] = round(mp.stat().st_size/1024,1) if mp.exists() else 0

# Server
try:
    resp = requests.get('http://localhost:8000/health', timeout=3)
    r['server_up'] = True
    r['server_ai_configured'] = resp.json().get('ai_configured')
except: r['server_up'] = False

# Review checks in isolation
from app.review.style import analyze_style
from app.review.security import analyze_security
from app.review.smells import analyze_smells
from app.review.complexity import analyze_complexity

code = "import os\ndef bad(list, data=[]):\n    password='abc123'\n    for i in range(10):\n        for j in range(10):\n            for k in range(10):\n                eval('x')\n    unused = 42\n    return data\n"
r['style_issues']      = len(analyze_style(code))
r['security_issues']   = len(analyze_security(code))
r['smell_issues']      = len(analyze_smells(code))
r['complexity_issues'] = len(analyze_complexity(code))

smells = analyze_smells(code)
r['smell002_fires'] = any(s.rule == 'SMELL002' for s in smells)
r['all_smell_rules'] = sorted(set(s.rule for s in smells))

# ML
from app.ml.complexity_predictor import _extract_features, predict_complexity
feat = _extract_features(code)
r['feature_vector_len'] = len(feat) if feat else 0

cases = {
    'O(1)':    'def f(lst): return lst[0]',
    'O(n)':    'def f(lst):\n    m=lst[0]\n    for x in lst:\n        if x>m: m=x\n    return m',
    'O(n2)':   'def f(lst):\n    for i in lst:\n        for j in lst:\n            pass',
    'O(n3)':   'def f(lst):\n    for i in lst:\n        for j in lst:\n            for k in lst:\n                pass',
    'O(nln)':  'def f(lst): return sorted(lst)',
    'O(2n)':   'def fib(n):\n    if n<=1: return n\n    return fib(n-1)+fib(n-2)',
    'fib_memo':'def fib(n,c={}):\n    if n in c: return c[n]\n    if n<=1: return n\n    c[n]=fib(n-1,c)+fib(n-2,c)\n    return c[n]',
    'seq_lps': 'def f(lst):\n    for x in lst: pass\n    for x in lst: pass',
}
preds = {}
for name, c in cases.items():
    p = predict_complexity(c, c)
    preds[name] = {'pred': p.before if p else 'NULL', 'conf': p.confidence if p else 0}
r['ml_predictions'] = preds

# Refactor
from app.refactor.orchestrator import run_all_rules
from app.models import OptimizationLevel
rr = run_all_rules('def f():\n    x = 2 * 8\n    unused = 99\n    if x == True:\n        return x\n    else:\n        return 0\n', level=OptimizationLevel.HIGH)
r['refactor_rules'] = [x.rule for x in rr.applied_rules]
r['refactored'] = rr.refactored_code

# Tests
import subprocess
tp = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-q', '--tb=no'],
                   capture_output=True, text=True, cwd=str(BASE_DIR))
last = [l for l in (tp.stdout+tp.stderr).splitlines() if l.strip()]
r['tests'] = last[-1] if last else 'unknown'

(BASE_DIR / 'status.json').write_text(json.dumps(r, indent=2), encoding='utf-8')
print('done')
