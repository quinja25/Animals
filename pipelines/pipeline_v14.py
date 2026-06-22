"""V14 launcher: boundary reconstruction, B/C specialist, and drift resistance."""

import os
import runpy
from pathlib import Path

os.environ.setdefault("PIPELINE_VERSION", "v14")
os.environ.setdefault("ENABLE_EXTENDED_LINEAGE", "1")
os.environ.setdefault("ENABLE_B_SPECIALIST", "0")
os.environ.setdefault("ENABLE_YIELD_BALANCE", "0")
os.environ.setdefault("ENABLE_DUAL_C", "1")
os.environ.setdefault("ENABLE_PROXY_TRAITS", "1")
os.environ.setdefault("ENABLE_BOUNDARY_PROXY", "1")
os.environ.setdefault("ENABLE_BC_SPECIALIST", "1")
os.environ.setdefault("ENABLE_DRIFT_RESISTANT", "1")
os.environ.setdefault("ENABLE_REGIONAL_CALIBRATION", "1")
os.environ.setdefault("C_BALANCE_POWER", "0.25")
os.environ.setdefault("PIPELINE_SMOKE_STAGE", os.getenv("V14_SMOKE_STAGE", ""))

runpy.run_path(Path(__file__).with_name("pipeline_v11.py"), run_name="__main__")
