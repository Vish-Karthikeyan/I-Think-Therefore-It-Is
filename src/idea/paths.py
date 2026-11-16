from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
# Repository root (…/idea). When installed non-editable, set IDEA_DATA_DIR.
REPO_DIR = PKG_DIR.parent.parent
DATA_DIR = Path(os.environ.get("IDEA_DATA_DIR", REPO_DIR / "data"))
RUNS_DIR = Path(os.environ.get("IDEA_RUNS_DIR", REPO_DIR / "runs"))
