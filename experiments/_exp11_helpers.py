"""Test-only import helpers for Experiment 11."""
from pathlib import Path
import runpy

ns = runpy.run_path(str(Path(__file__).resolve().parents[1] / "experiments/11_multipath_fading_receiver.py"))
memory_filter = ns["_memory_filter"]
channel = ns["_channel"]
