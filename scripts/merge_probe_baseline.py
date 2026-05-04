"""Merge probe-baseline shards (A/B/C) and report cross-family headline.

Reads:  results/probe_steering/probe_baseline_gemma-2-2b_controlled_{A,B,C}.json
Writes: results/probe_steering/probe_baseline_gemma-2-2b_controlled.json
        (merged across all 25 families)

Prints a headline summary: per-family table + cross-family aggregates suitable
for the App. D paragraph in the paper.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results/probe_steering"


def merge():
    out = {"config": None, "families": {}}
    for tag in "ABC":
        p = RES / f"probe_baseline_gemma-2-2b_controlled_{tag}.json"
        d = json.load(open(p))
        if out["config"] is None:
            out["config"] = d["config"]
        for k, v in d["families"].items():
            assert k not in out["families"], f"duplicate family {k}"
            out["families"][k] = v
    merged_path = RES / "probe_baseline_gemma-2-2b_controlled.json"
    with open(merged_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Merged {len(out['families'])} families -> {merged_path.name}")
    return out


def isfinite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def ci_percentile(values, alpha=0.05, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray([v for v in values if isfinite(v)], dtype=float)
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    out = merge()
    fams = out["families"]

    print("\n" + "=" * 90)
    print(f"{'family':18s} {'mean_acc':>9s} {'max_dp_p':>9s} {'max_dp_m':>9s} "
          f"{'max_dp_r':>9s} {'rho_p,m':>8s} {'rho_acc,p':>10s}")
    print("-" * 90)
    rows = []
    for fam in sorted(fams):
        s = fams[fam]["summary"]
        rho_pm = s["rho(delta_p_mds, delta_p_probe)"]
        rho_acc_p = s["rho(probe_acc, delta_p_probe)"]
        rows.append((fam, s["mean_probe_acc"], s["max_dp_probe"],
                     s["max_dp_mds"], s["max_dp_random"], rho_pm, rho_acc_p))
        print(f"{fam:18s} {s['mean_probe_acc']:9.3f} "
              f"{s['max_dp_probe']:9.4f} {s['max_dp_mds']:9.4f} "
              f"{s['max_dp_random']:9.4f} "
              f"{rho_pm if isfinite(rho_pm) else float('nan'):8.3f} "
              f"{rho_acc_p if isfinite(rho_acc_p) else float('nan'):10.3f}")

    print("=" * 90)
    n = len(rows)
    mean_acc = np.mean([r[1] for r in rows])
    median_max_dp_probe = np.median([r[2] for r in rows])
    median_max_dp_mds = np.median([r[3] for r in rows])
    median_max_dp_random = np.median([r[4] for r in rows])

    rho_pm_vals = [r[5] for r in rows if isfinite(r[5])]
    rho_acc_p_vals = [r[6] for r in rows if isfinite(r[6])]

    rho_pm_med = float(np.median(rho_pm_vals)) if rho_pm_vals else float("nan")
    rho_pm_ci = ci_percentile(rho_pm_vals)
    rho_acc_p_med = (float(np.median(rho_acc_p_vals))
                     if rho_acc_p_vals else float("nan"))
    rho_acc_p_ci = ci_percentile(rho_acc_p_vals)

    n_undef = sum(1 for r in rows if not isfinite(r[6]))

    print(f"\n=== Cross-family aggregate (n={n}) ===")
    print(f"  mean probe accuracy across families: {mean_acc:.3f}")
    print(f"  median max ΔP_probe : {median_max_dp_probe:.4f}")
    print(f"  median max ΔP_MDS   : {median_max_dp_mds:.4f}")
    print(f"  median max ΔP_rand  : {median_max_dp_random:.4f}")
    print(f"  median ρ(ΔP_MDS, ΔP_probe) [per-family]      : "
          f"{rho_pm_med:.3f} (mean-of-resamples 95% CI {rho_pm_ci[0]:.2f}, {rho_pm_ci[1]:.2f}; "
          f"n_finite={len(rho_pm_vals)})")
    print(f"  median ρ(probe_acc, ΔP_probe) [per-family]   : "
          f"{rho_acc_p_med:.3f} (mean-of-resamples 95% CI {rho_acc_p_ci[0]:.2f}, {rho_acc_p_ci[1]:.2f}; "
          f"n_finite={len(rho_acc_p_vals)}, n_undef={n_undef})")

    pooled_acc = []
    pooled_dp_probe = []
    pooled_dp_mds = []
    for fam in sorted(fams):
        for row in fams[fam]["rows"]:
            if isfinite(row.get("probe_train_acc")) and isfinite(row.get("delta_p_probe")):
                pooled_acc.append(row["probe_train_acc"])
                pooled_dp_probe.append(row["delta_p_probe"])
                pooled_dp_mds.append(row["delta_p_mds"])
    if pooled_acc:
        rho_acc_dp_pool = spearmanr(pooled_acc, pooled_dp_probe).correlation
        rho_mds_probe_pool = spearmanr(pooled_dp_mds, pooled_dp_probe).correlation
        print(f"\n=== Pooled across (family, layer) cells (n={len(pooled_acc)}) ===")
        print(f"  ρ(probe_acc, ΔP_probe)  = {rho_acc_dp_pool:+.3f}")
        print(f"  ρ(ΔP_MDS, ΔP_probe)     = {rho_mds_probe_pool:+.3f}")

    print("\n=== Per-family undefined ρ(probe_acc, ΔP_probe) ===")
    for r in rows:
        if not isfinite(r[6]):
            print(f"  {r[0]} (mean_acc={r[1]:.3f})")


if __name__ == "__main__":
    main()
