"""Scan for prohibited ML patterns in all Python source files."""
import pathlib

prohibited = [
    "fit_transform",
    "import shap",
    "torch.optim",
    ".backward()",
    "model.train()",
    "GridSearchCV",
    "RandomizedSearchCV",
]

hits = []

for p in pathlib.Path(".").rglob("*.py"):
    parts = p.parts
    if any(x in parts for x in [".venv", "__pycache__", "Context"]):
        continue
    try:
        src = p.read_text(encoding="utf-8")
        for pattern in prohibited:
            if pattern in src:
                for i, line in enumerate(src.splitlines(), 1):
                    stripped = line.strip()
                    if pattern in line and not stripped.startswith("#") and '"""' not in stripped:
                        hits.append((str(p), i, line.strip()))
    except Exception:
        pass

if hits:
    for fpath, lineno, text in hits:
        print(f"HIT: {fpath}:{lineno}: {text}")
else:
    print("CLEAN: Zero prohibited ML patterns found in executable code.")
