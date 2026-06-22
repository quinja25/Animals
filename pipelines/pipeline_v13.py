"""V13 launcher: nested carcass reconstruction plus dual-C yield ensemble."""

import os
import runpy
from pathlib import Path

os.environ.setdefault("PIPELINE_VERSION", "v13")
os.environ.setdefault("ENABLE_EXTENDED_LINEAGE", "1")
os.environ.setdefault("ENABLE_B_SPECIALIST", "0")
os.environ.setdefault("ENABLE_YIELD_BALANCE", "0")
os.environ.setdefault("ENABLE_DUAL_C", "1")
os.environ.setdefault("ENABLE_PROXY_TRAITS", "1")
os.environ.setdefault("C_BALANCE_POWER", "0.25")
os.environ.setdefault("PIPELINE_SMOKE_STAGE", os.getenv("V13_SMOKE_STAGE", ""))

runpy.run_path(Path(__file__).with_name("pipeline_v11.py"), run_name="__main__")
