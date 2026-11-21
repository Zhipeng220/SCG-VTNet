"""Paths for research inputs kept outside the public repository."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("SCG_VTNET_WORKSPACE", Path.cwd().parent)).expanduser().resolve()
ASL_IMAGES = ROOT / "data/raw/asl_alphabet"
ASL_PROCESSED = ROOT / "data/processed/asl_alphabet"
ASL_SPLITS = ROOT / "data/splits/asl_alphabet"
RUNS = ROOT / "runs"
