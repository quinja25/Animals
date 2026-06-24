"""V12 launcher for the shared specialized boosting engine."""

import os
import runpy
from pathlib import Path

os.environ.setdefault("PIPELINE_VERSION", "v12")
os.environ.setdefault("ENABLE_EXTENDED_LINEAGE", "1")
os.environ.setdefault("ENABLE_B_SPECIALIST", "1")
os.environ.setdefault("ENABLE_YIELD_BALANCE", "1")
os.environ.setdefault("C_BALANCE_POWER", "0.25")
os.environ.setdefault("PIPELINE_SMOKE_STAGE", os.getenv("V12_SMOKE_STAGE", ""))

runpy.run_path(Path(__file__).with_name("pipeline_v11.py"), run_name="__main__")
