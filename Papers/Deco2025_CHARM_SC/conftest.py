"""
conftest.py
-----------
Pytest configuration for the Deco2025_CHARM_SC project.

This file does two things:

1. Its presence at the project root tells pytest to add this directory
   to sys.path, so that `from geometry import ...`, `from simulation import ...`
   etc. all resolve correctly from any test file.

2. It ensures neuronumba is importable by pytest. When neuronumba is
   installed via `pip install -e` pointing at a git repo, the editable
   install may not be visible to pytest's Python environment depending
   on how pytest was invoked. We add the neuronumba src path explicitly
   as a fallback so tests never fail due to a path issue rather than a
   real code problem.

Usage
-----
    python -m pytest tests/ -v          # from project root
    python -m pytest tests/ -v -s       # with stdout (useful for debugging)
"""

import sys
import subprocess
from pathlib import Path


# ── 1. Ensure the project root is on sys.path ─────────────────────────────────
# conftest.py's location IS the project root — add it explicitly so
# submodule imports work regardless of how pytest was invoked.
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── 2. Ensure neuronumba is importable ───────────────────────────────────────
# neuronumba is installed via `pip install -e git+...` which registers it
# in site-packages of the *installing* Python. If pytest is run with a
# different Python context (e.g. `python -m pytest`), it may not see it.
# We locate it via `pip show` and add its src to sys.path as a fallback.
try:
    import neuronumba  # noqa: F401 — check if already importable
except ModuleNotFoundError:
    try:
        # Ask pip where neuronumba is installed
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', '-f', 'neuronumba'],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith('Location:'):
                location = line.split(':', 1)[1].strip()
                # editable installs from a subdirectory (subdirectory=src)
                # are registered at the src/ level
                for candidate in [
                    location,
                    str(Path(location) / 'src'),
                    str(Path(location).parent / 'src'),
                ]:
                    if Path(candidate).is_dir() and candidate not in sys.path:
                        sys.path.insert(0, candidate)
    except Exception:
        pass  # best-effort — the fallback FC in bold_generator.py handles the rest
