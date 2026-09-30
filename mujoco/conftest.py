"""Root conftest.py — adds project sub-directories to sys.path.

controllers/ and simulations/ contain modules that import each other by bare
name (e.g. ``from prc import ...``).  Adding both directories to
sys.path lets Python resolve those imports without requiring the packages to
be installed or the import statements to be changed.
"""

import sys
from pathlib import Path

_root = Path(__file__).parent

for _sub in ("controllers", "simulations"):
    _p = str(_root / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
