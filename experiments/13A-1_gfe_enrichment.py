"""Post-process 13A-1 results with GFE/GGFE spectral-memory metrics.

This does not rerun the 13A-1 simulation. It derives kernel-level metrics
directly from the recorded SOE parameters (weights and decay rates), then
joins them to the existing 675-row atlas.

Definitions follow the GGFE/Spectral Memory Units formulation:
M_cap = (sum w/gamma^2) / (sum w/gamma)
M_scale = log10(gamma_max/gamma_min)
M_res = L/M_scale
H_mem = -sum w_tilde log(w_tilde)
D_eff = L*exp(H_mem)

The entropy-effective component count exp(H_mem) is also retained separately
to distinguish weight-distribution complexity from the documented D_eff.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
INPUT = RESULTS / "13A-1_kernel_soe_geometry_atlas_results.csv"
OUTPUT = RESULTS / "13A-1_kernel_soe_geometry_atlas_gfe_results.csv"
SUMMARY = RESULTS / "13A-1_kernel_soe_geometry_atlas_gfe_summary.json"


def spectral_metrics(row: dict) -> dict:
    L = int(row["mode_count"])
    gamma_min = float(row["gamma_min"])
    gamma_max = float(row["gamma_max"])
    w_min = float(row["weight_min"])
    w_max = float(row["weight_max"])

    # The atlas records only min/max weights, not the complete vector. Recover
    # the exact deterministic weight vector from the named pattern.
    if L == 1:
        weights = np.ones(1, dtype=float)
    elif row["weight_pattern"] == "uniform":
        weights = np.full(L, 1.0 / L)
    else:
        weights = np.full(L, 0.3 / (L - 1), dtype=float)
        weights[0 if row["weight_pattern"] == "slow_dominant" else -1] = 0.7

    # Recover the exact deterministic gamma grid from the atlas condition.
    geometry = row["rate_geometry"]
    if geometry == "clustered":
        gammas = 10.0 * np.exp(np.linspace(-0.05, 0.05, L))
    elif geometry == "logspread":
        gammas = np.geomspace(2.0, 30.0, L)
    elif geometry == "wide":
        gammas = np.geomspace(0.5, 100.0, L)
    else:
        raise ValueError(f"Unknown rate geometry: {geometry}")

    if np.any(weights <= 0) or np.any(gammas <= 0):
        raise ValueError("GFE spectral units require positive weights and rates.")

    wt = weights / np.sum(weights)
    m_cap = float(np.sum(weights / gammas**2) / np.sum(weights / gammas))
    m_scale = float(math.log10(np.max(gammas) / np.min(gammas)))
    m_res = float(L / m_scale) if m_scale > 0 else float("inf")
    h_mem = float(-np.sum(wt * np.log(wt)))
    entropy_effective_count = float(np.exp(h_mem))
    d_eff = float(L * entropy_effective_count)

    return {
        "gfe_m_cap_s": m_cap,
        "gfe_m_scale_decades": m_scale,
        "gfe_m_res_modes_per_decade": m_res,
        "gfe_h_mem_nats": h_mem,
        "gfe_entropy_effective_count": entropy_effective_count,
        "gfe_d_eff": d_eff,
    }


def main() -> None:
    with INPUT.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 675:
        raise ValueError(f"Expected the audited 13A-1 atlas to contain 675 rows; got {len(rows)}.")

    enriched = []
    for row in rows:
        row = dict(row)
        row.update(spectral_metrics(row))
        enriched.append(row)

    fields = list(enriched[0])
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(enriched)

    names = [
        "gfe_m_cap_s",
        "gfe_m_scale_decades",
        "gfe_m_res_modes_per_decade",
        "gfe_h_mem_nats",
        "gfe_entropy_effective_count",
        "gfe_d_eff",
        "state_participation_dimension",
        "state_weighted_participation_dimension",
        "state_max_component_collinearity",
        "kernel_t90_s",
        "kernel_centroid_s",
    ]

    matrix = np.array([[float(r[n]) for n in names] for r in enriched], dtype=float)
    corr = np.corrcoef(matrix, rowvar=False)
    correlations = {}
    for i, name in enumerate(names[:6]):
        correlations[name] = {
            other: float(corr[i, j])
            for j, other in enumerate(names[6:], start=6)
        }

    summary = {
        "experiment": "13A-1_kernel_soe_geometry_atlas",
        "analysis": "GFE/GGFE spectral-memory enrichment",
        "base_rows": len(rows),
        "rerun_simulation": False,
        "input": str(INPUT.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
        "definitions": {
            "M_cap": "sum(w/gamma^2) / sum(w/gamma)",
            "M_scale": "log10(gamma_max/gamma_min)",
            "M_res": "L/M_scale",
            "H_mem": "-sum(w_tilde*ln(w_tilde))",
            "entropy_effective_count": "exp(H_mem)",
            "D_eff": "L*exp(H_mem), following GGFE/Spectral Memory Units",
        },
        "interpretation": [
            "These metrics characterize the SOE kernel, not the empirical state trajectory.",
            "D_eff is retained as the documented GGFE quantity; exp(H_mem) is retained separately.",
            "M_res is secondary because clustered rates can be spectrally dense yet highly redundant.",
            "No d_s is assigned to this finite positive SOE atlas; spectral/fractal dimension belongs to later non-Weyl or continuous-spectrum experiments.",
        ],
        "kernel_to_state_correlations_pearson": correlations,
    }

    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
