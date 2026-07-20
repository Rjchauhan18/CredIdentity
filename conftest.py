"""Pytest bootstrap: put the repo root on sys.path.

The project uses namespace packages (no __init__.py), and modules are invoked as
`backend.main:app` / `python data_pipeline/engineer_features.py` from the repo root.
Adding the root here lets tests import `data_pipeline.*`, `backend.*`, etc. the same
way the app does.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
