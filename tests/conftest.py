"""Shared pytest fixtures.

Adds the project root to sys.path so `import config`, `from tools import ...`,
etc. work when pytest is invoked from anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
