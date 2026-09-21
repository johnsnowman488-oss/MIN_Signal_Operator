"""Small test helper imported by Experiment 10 tests."""

from pathlib import Path
import runpy

_namespace = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "experiments" / "10_awgn_receiver_equalization.py")
)
apply_min_uniform = _namespace["apply_min_uniform"]
