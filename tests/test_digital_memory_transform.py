import importlib.util
import sys
from pathlib import Path
import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "experiments" / "09_digital_memory_transform.py"
spec = importlib.util.spec_from_file_location("experiment09", MODULE_PATH)
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)


def test_identity_has_zero_evm():
    signal = mod.SIGNALS["QPSK"](64, samples_per_symbol=8, seed=3)
    observed = signal.record.samples[signal.symbol_indices]
    assert np.isclose(mod.evm_after_gain(signal.symbols, observed), 0.0, atol=1e-12)


def test_memory_output_is_finite():
    signal = mod.SIGNALS["16QAM"](64, samples_per_symbol=8, seed=3)
    t = np.arange(signal.record.samples.size) / signal.record.sample_rate
    y = mod.apply_kernel("powerlaw_alpha_0.70", t, signal.record.samples)
    assert np.isfinite(y).all()


def test_experiment_produces_expected_row_count():
    rows=[]
    for seed in range(2):
        for name,generator in mod.SIGNALS.items():
            rows.extend(mod.run_one(name,generator,seed,sps=8))
    assert len(rows)==2*len(mod.SIGNALS)*len(mod.KERNELS)
    assert all(r["evm_percent"]>=0 for r in rows)
