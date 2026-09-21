"""Analyze Experiment 12A receiver-recoverability results.

Usage:
    python experiments/12A_receiver_recoverability_analysis.py

Produces aggregate CSV files and PNG figures when matplotlib is installed.
The script does not rank receivers globally; it exposes complexity/error
trade-offs by memory, channel, SNR, and signal.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results/12_receiver_recoverability_results.csv"
OUT = ROOT / "experiments/results"


def _read_rows():
    with RESULTS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    numeric = (
        "snr_db", "seed", "evm_percent", "ber", "heldout_mse",
        "parameter_count", "state_dimension", "macs_per_sample",
        "effective_memory_samples", "pole_radius",
    )
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def _group(rows, keys):
    buckets = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        buckets.setdefault(key, []).append(row)
    return buckets


def _write_summary(rows, name, keys):
    path = OUT / name
    fields = list(keys) + [
        "n", "mean_evm_percent", "std_evm_percent",
        "mean_ber", "std_ber", "mean_heldout_mse",
        "mean_pole_radius",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, group in sorted(_group(rows, keys).items()):
            evm = np.array([r["evm_percent"] for r in group])
            ber = np.array([r["ber"] for r in group])
            mse = np.array([r["heldout_mse"] for r in group])
            poles = np.array([r["pole_radius"] for r in group])
            w.writerow(dict(zip(keys, key)) | {
                "n": len(group),
                "mean_evm_percent": float(evm.mean()),
                "std_evm_percent": float(evm.std(ddof=0)),
                "mean_ber": float(ber.mean()),
                "std_ber": float(ber.std(ddof=0)),
                "mean_heldout_mse": float(mse.mean()),
                "mean_pole_radius": float(np.nanmean(poles)),
            })
    return path


def _plot(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; aggregate CSV files were still produced.")
        return

    # Collapse over signal/seed for readable complexity curves, retaining
    # memory, channel, and SNR as the experimental condition.
    keys = ("memory", "channel", "snr_db", "receiver")
    grouped = _group(rows, keys)
    points = []
    for key, group in grouped.items():
        points.append({
            **dict(zip(keys, key)),
            "state_dimension": group[0]["state_dimension"],
            "macs_per_sample": group[0]["macs_per_sample"],
            "evm": float(np.mean([r["evm_percent"] for r in group])),
            "ber": float(np.mean([r["ber"] for r in group])),
        })

    conditions = sorted({(p["memory"], p["channel"], p["snr_db"]) for p in points})
    for metric, ylabel, filename in (
        ("evm", "Mean held-out EVM (%)", "12A_evm_vs_state.png"),
        ("ber", "Mean held-out BER", "12A_ber_vs_state.png"),
    ):
        for memory, channel, snr in conditions:
            subset = [
                p for p in points
                if p["memory"] == memory and p["channel"] == channel and p["snr_db"] == snr
            ]
            subset.sort(key=lambda p: (p["state_dimension"], p["receiver"]))
            x = [p["state_dimension"] for p in subset]
            y = [p[metric] for p in subset]
            labels = [p["receiver"] for p in subset]
            plt.figure(figsize=(7, 4.5))
            plt.plot(x, y, marker="o")
            for xi, yi, label in zip(x, y, labels):
                plt.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points")
            plt.xlabel("Receiver state dimension")
            plt.ylabel(ylabel)
            plt.title(f"{memory} | {channel} | {snr:g} dB")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            safe = f"{memory}_{channel}_{snr:g}dB".replace(".", "p")
            plt.savefig(OUT / f"{safe}_{filename}", dpi=160)
            plt.close()

    # Complexity plane: every receiver family is shown under each condition.
    for memory, channel, snr in conditions:
        subset = [
            p for p in points
            if p["memory"] == memory and p["channel"] == channel and p["snr_db"] == snr
        ]
        subset.sort(key=lambda p: p["macs_per_sample"])
        plt.figure(figsize=(7, 4.5))
        plt.plot([p["macs_per_sample"] for p in subset],
                 [p["evm"] for p in subset], marker="o")
        for p in subset:
            plt.annotate(p["receiver"], (p["macs_per_sample"], p["evm"]),
                         xytext=(4, 4), textcoords="offset points")
        plt.xlabel("MACs/sample (accounting convention)")
        plt.ylabel("Mean held-out EVM (%)")
        plt.title(f"EVM vs arithmetic | {memory} | {channel} | {snr:g} dB")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        safe = f"{memory}_{channel}_{snr:g}dB".replace(".", "p")
        plt.savefig(OUT / f"{safe}_12A_evm_vs_macs.png", dpi=160)
        plt.close()


def main():
    if not RESULTS.exists():
        raise SystemExit(f"Missing result file: {RESULTS}. Run 12_receiver_recoverability.py first.")
    rows = _read_rows()
    _write_summary(rows, "12_receiver_recoverability_by_condition.csv",
                   ("memory", "channel", "snr_db", "receiver"))
    _write_summary(rows, "12_receiver_recoverability_by_signal.csv",
                   ("signal", "memory", "channel", "snr_db", "receiver"))
    _plot(rows)
    print(f"Analyzed {len(rows)} rows.")


if __name__ == "__main__":
    main()
