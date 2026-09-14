"""Puts the repo root on sys.path so tests can import scripts/ as a package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
