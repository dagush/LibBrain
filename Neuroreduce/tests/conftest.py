# LibBrain/Neuroreduce/tests/conftest.py
# ----------------------------------------
# Pytest path configuration for Neuroreduce tests.
#
# Problem this solves
# --------------------
# Neuroreduce lives at LibBrain/Neuroreduce/. When pytest is run from
# inside LibBrain/Neuroreduce/tests/, Python adds only tests/ to sys.path.
# That makes `from Neuroreduce import ...` fail because Neuroreduce/ is
# one level above tests/, and two levels above LibBrain/.
#
# Solution
# ---------
# This conftest.py is discovered by pytest before any test is collected.
# It adds LibBrain/ (the root that CONTAINS the Neuroreduce folder) to
# sys.path, so `from Neuroreduce import CHARMReducer` works identically
# to how it works when running scripts from LibBrain/ directly.
#
# Directory layout
# -----------------
# LibBrain/                        ← LIBBRAIN_ROOT (added to sys.path)
#   Neuroreduce/
#     methods/
#     utils/
#     tests/
#       conftest.py                ← this file  (Path(__file__))
#       test_charm_sc.py
#       test_pca.py
#       ...
#
# Usage
# ------
# Run from anywhere:
#     cd LibBrain/Neuroreduce/tests
#     python -m pytest test_charm_sc.py -v
#
# Or from LibBrain root:
#     python -m pytest Neuroreduce/tests/ -v

import sys
from pathlib import Path

# __file__ = LibBrain/Neuroreduce/tests/conftest.py
# .parent        = LibBrain/Neuroreduce/tests/
# .parent.parent = LibBrain/Neuroreduce/
# .parent.parent.parent = LibBrain/            ← this is what we need
LIBBRAIN_ROOT = Path(__file__).resolve().parent.parent.parent

if str(LIBBRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBBRAIN_ROOT))
