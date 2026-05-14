"""Entry point for Img-Tagboru desktop application.

Run with:
    python run.py
or (no console window):
    pythonw run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path before any local imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.native.main_window import main

if __name__ == "__main__":
    main()
