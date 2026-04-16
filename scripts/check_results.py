"""Compare results in ./results/ against the published paper baseline.

Re-derives headline statistics from each experiment's JSON output and prints a
PASS/FAIL line per check. Tolerances are set to absorb stochasticity of
probe training and steering CV (probes ~0.02 abs, deltaP ~0.05 abs,
correlations ~0.05 abs).

Usage:
    python scripts/check_results.py
    python scripts/check_results.py --strict   # tighter tolerances
"""

import scripts._env  # noqa: F401

import argparse
import json
import sys
from pathlib import Path
from scipy.stats import spearmanr

R = Path("results")

# (key, expected, abs_tolerance)
# Values are from the regression run that backs the paper as submitted.
EXPECTED = {
    "exp1_per_family": {
        # peak A_lin per family (Gemma-2-2B)
        "arithmetic.peak_alin": (0.686, 0.03),
        "geography.peak_alin":  (0.280, 0.03),
        "sequence.peak_alin":   (0.711, 0.03),
        "word_transform.peak_alin": (0.512, 0.03),
        "analogy.peak_alin":    (0.451, 0.03),
        # peak A_mlp per family
        "arithmetic.peak_amlp": (0.904, 0.04),
        "geography.peak_amlp":  (1.000, 0.02),
        "sequence.peak_amlp":   (0.954, 0.04),
        "word_transform.peak_amlp": (0.647, 0.05),
        "analogy.peak_amlp":    (0.629, 0.05),
    },
    "exp2_per_family_steering": {
        # max ΔP per family (Gemma-2-2B, across layers)
        "arithmetic.max_dp": (0.246, 0.05),
        "geography.max_dp":  (0.238, 0.05),
        "sequence.max_dp":   (0.157, 0.05),
        "word_transform.max_dp": (0.386, 0.05),
        "analogy.max_dp":    (0.191, 0.05),
        # per-family rho(A_lin, ΔP) across layers
        "arithmetic.rho_alin_dp": (0.719, 0.08),
        "geography.rho_alin_dp":  (0.758, 0.08),
        "sequence.rho_alin_dp":   (0.814, 0.08),
        "word_transform.rho_alin_dp": (0.871, 0.08),
        "analogy.rho_alin_dp":    (0.866, 0.08),
        # pooled
        "pooled.rho_alin_dp": (0.777, 0.05),
    },
    "cross_concept": {
        # rho across controlled families (peak A_lin vs max ΔP)
        "Pythia-2.8B.all": (0.89, 0.05),
        "Pythia-2.8B.controlled": (0.86, 0.05),
        "Gemma-2-2B.all": (0.86, 0.05),
        "Gemma-2-2B.controlled": (0.86, 0.05),
        "Qwen2.5-1.5B.all": (0.90, 0.05),
        "Qwen2.5-1.5B.controlled": (0.86, 0.05),
        "Qwen2.5-7B.all": (0.92, 0.05),
        "Qwen2.5-7B.controlled": (0.90, 0.05),
        "Llama-3.1-8B.all": (0.93, 0.05),
        "Llama-3.1-8B.controlled": (0.91, 0.05),
    },
    "scaling": {
        # Pythia size -> (mean A_lin, mean ΔP, n_steerable, rho)
        "160m.mean_alin": (0.055, 0.02),
        "160m.mean_dp":   (0.011, 0.02),
        "160m.n_steerable": (5, 1),
        "160m.rho":       (0.42, 0.10),
        "410m.mean_alin": (0.075, 0.02),
        "410m.n_steerable": (6, 1),
        "1b.mean_alin":   (0.087, 0.02),
        "1b.n_steerable": (7, 1),
        "2.8b.mean_alin": (0.111, 0.02),
        "2.8b.n_steerable": (9, 1),
        "2.8b.rho":       (0.86, 0.06),
        "6.9b.mean_alin": (0.118, 0.02),
        "6.9b.n_steerable": (9, 1),
        "6.9b.rho":       (0.86, 0.06),
    },
    "replication": {
        # peak A_lin per family per model
        "Llama-3.1-8B.arithmetic.peak_alin": (0.995, 0.03),
        "Llama-3.1-8B.geography.peak_alin":  (0.680, 0.05),
        "Llama-3.1-8B.sequence.peak_alin":   (0.820, 0.04),
        "Mistral-7B-v0.3.arithmetic.peak_alin": (0.695, 0.05),
        "Mistral-7B-v0.3.geography.peak_alin":  (0.540, 0.05),
        "Mistral-7B-v0.3.sequence.peak_alin":   (0.735, 0.05),
        "Qwen2.5-7B.arithmetic.peak_alin": (0.935, 0.03),
        "Qwen2.5-7B.geography.peak_alin":  (0.585, 0.05),
        "Qwen2.5-7B.sequence.peak_alin":   (0.780, 0.04),
        # per-family rho across layers
        "Llama-3.1-8B.arithmetic.rho": (0.849, 0.08),
        "Llama-3.1-8B.geography.rho":  (0.930, 0.08),
        "Llama-3.1-8B.sequence.rho":   (0.904, 0.08),
        "Mistral-7B-v0.3.arithmetic.rho": (0.886, 0.08),
        "Mistral-7B-v0.3.geography.rho":  (0.785, 0.10),
        "Mistral-7B-v0.3.sequence.rho":   (0.885, 0.08),
        "Qwen2.5-7B.arithmetic.rho": (0.707, 0.10),
        "Qwen2.5-7B.geography.rho":  (0.717, 0.10),
        "Qwen2.5-7B.sequence.rho":   (0.657, 0.10),
    },
    "demos": {
        "gemma_entity.rho":  (0.66, 0.08),
        "olmo_entity.rho":   (0.74, 0.08),
        "refusal.rho":       (0.91, 0.05),
    },
}

STRICT_SCALE = 0.5  # multiply tolerances by this when --strict


def fmt(v):
    if isinstance(v, int):
        return str(v)
    return f"{v:+.3f}"


def check(actual, expected_pair, strict=False):
    """Return (ok, line_text). Tolerance is the second element of expected_pair."""
    expected, tol = expected_pair
    if strict:
        tol = tol * STRICT_SCALE
    if actual is None:
        return False, f"MISSING (expected {fmt(expected)})"
    diff = abs(actual - expected)
    ok = diff <= tol
    badge = "PASS" if ok else "FAIL"
    return ok, f"{badge}  got={fmt(actual)}  expected={fmt(expected)} (±{tol:g})"


# ---------- per-experiment computation ----------

def compute_exp1_per_family():
    out = {}
    f1 = R / "exp1" / "layer_accuracy.json"
    f2 = R / "exp1" / "mlp_probe_results.json"
    if not f1.exists() or not f2.exists():
        return None
    with open(f1) as f:
        lin = json.load(f)
    with open(f2) as f:
        mlp = json.load(f)
    for fam in ["arithmetic", "geography", "sequence", "word_transform", "analogy"]:
        if fam not in lin:
            continue
        lin_layers = {int(l): lin[fam]["layers"][l]["linear_acc"] for l in lin[fam]["layers"]}
        out[f"{fam}.peak_alin"] = max(lin_layers.values())
        if fam in mlp:
            mlp_layers = {int(l): mlp[fam][l]["mlp_acc"] for l in mlp[fam]}
            out[f"{fam}.peak_amlp"] = max(mlp_layers.values())
    return out


def compute_exp2_per_family():
    out = {}
    f1 = R / "exp1" / "layer_accuracy.json"
    f2 = R / "exp2" / "steering_results.json"
    if not f1.exists() or not f2.exists():
        return None
    with open(f1) as f:
        lin = json.load(f)
    with open(f2) as f:
        steer = json.load(f)
    all_a, all_d = [], []
    for fam in ["arithmetic", "geography", "sequence", "word_transform", "analogy"]:
        if fam not in lin or fam not in steer:
            continue
        lin_layers = {int(l): lin[fam]["layers"][l]["linear_acc"] for l in lin[fam]["layers"]}
        steer_layers = {int(l): steer[fam]["layers"][l]["mean_delta_p"] for l in steer[fam]["layers"]}
        out[f"{fam}.max_dp"] = max(steer_layers.values())
        layers = sorted(set(lin_layers) & set(steer_layers))
        a = [lin_layers[l] for l in layers]
        d = [steer_layers[l] for l in layers]
        rho, _ = spearmanr(a, d)
        out[f"{fam}.rho_alin_dp"] = rho
        all_a.extend(a)
        all_d.extend(d)
    if all_a:
        out["pooled.rho_alin_dp"] = spearmanr(all_a, all_d)[0]
    return out


CC_DIRS = {
    "Pythia-2.8B": "pythia-2.8b-deduped",
    "Gemma-2-2B": "gemma-2-2b",
    "Qwen2.5-1.5B": "Qwen2.5-1.5B",
    "Qwen2.5-7B": "Qwen2.5-7B",
    "Llama-3.1-8B": "Llama-3.1-8B",
}


def compute_cross_concept():
    out = {}
    base = R / "cross_concept"
    if not base.exists():
        return None
    for nice, dirname in CC_DIRS.items():
        p = base / dirname / "results.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        fams = d["families"]
        all_pairs = [(f["best_alin"], f["max_steering_dp"])
                     for f in fams.values() if f.get("max_steering_dp") is not None]
        if all_pairs:
            out[f"{nice}.all"] = spearmanr([a for a, _ in all_pairs],
                                           [d for _, d in all_pairs])[0]
        ctrl_pairs = [(f["best_alin"], f["max_steering_dp"])
                      for n, f in fams.items()
                      if n.startswith("c_") and f.get("max_steering_dp") is not None]
        if ctrl_pairs:
            out[f"{nice}.controlled"] = spearmanr([a for a, _ in ctrl_pairs],
                                                  [d for _, d in ctrl_pairs])[0]
    return out


def compute_scaling():
    p = R / "scaling" / "scaling_results.json"
    if not p.exists():
        return None
    with open(p) as f:
        s = json.load(f)
    out = {}
    for sz in ["160m", "410m", "1b", "2.8b", "6.9b"]:
        if sz not in s:
            continue
        x = s[sz]
        out[f"{sz}.mean_alin"]   = x.get("mean_alin")
        out[f"{sz}.mean_dp"]     = x.get("mean_dp")
        out[f"{sz}.n_steerable"] = x.get("n_steerable")
        out[f"{sz}.rho"]         = x.get("rho_alin_dp")
    return out


REP_DIRS = {
    "Llama-3.1-8B":     "replication_Llama-3.1-8B",
    "Mistral-7B-v0.3":  "replication_Mistral-7B-v0.3",
    "Qwen2.5-7B":       "replication_Qwen2.5-7B",
}


def compute_replication():
    out = {}
    for nice, dirname in REP_DIRS.items():
        p = R / dirname / "results.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        fams = d.get("families", {})
        for fam in ["arithmetic", "geography", "sequence"]:
            if fam not in fams:
                continue
            fd = fams[fam]
            la = fd.get("layer_accuracy", {})
            st = fd.get("steering", {})
            if la:
                out[f"{nice}.{fam}.peak_alin"] = max(la.values())
            if la and st:
                layers = sorted(set(la) & set(st))
                if len(layers) >= 3:
                    a = [la[l] for l in layers]
                    dp = [st[l] for l in layers]
                    out[f"{nice}.{fam}.rho"] = spearmanr(a, dp)[0]
    return out


def compute_demos():
    out = {}
    for key, path in [
        ("gemma_entity.rho", R / "entity_steering_demo" / "entity_steering_results.json"),
        ("olmo_entity.rho",  R / "entity_steering_olmo" / "entity_steering_demo" / "entity_steering_results.json"),
        ("refusal.rho",      R / "refusal_demo" / "refusal_results.json"),
    ]:
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            out[key] = d.get("correlation_rho")
    return out


SECTIONS = [
    ("EXP1 — peak A_lin / A_mlp per family (Gemma-2-2B)",
     "exp1_per_family", compute_exp1_per_family),
    ("EXP2 — per-family steering (Gemma-2-2B)",
     "exp2_per_family_steering", compute_exp2_per_family),
    ("CROSS-CONCEPT — controlled-family rho per model",
     "cross_concept", compute_cross_concept),
    ("SCALING — Pythia 160M to 6.9B",
     "scaling", compute_scaling),
    ("REPLICATION — peak A_lin and within-family rho",
     "replication", compute_replication),
    ("DEMOS — entity steering and refusal",
     "demos", compute_demos),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="halve all tolerances")
    args = parser.parse_args()

    total = 0
    failed = 0
    missing_sections = []
    for title, expected_key, fn in SECTIONS:
        print("=" * 78)
        print(title)
        print("=" * 78)
        actual = fn()
        if actual is None:
            print(f"  (results not found — skipping {expected_key})")
            missing_sections.append(expected_key)
            continue
        for key, exp_pair in EXPECTED[expected_key].items():
            ok, msg = check(actual.get(key), exp_pair, strict=args.strict)
            total += 1
            if not ok:
                failed += 1
            print(f"  {key:<46} {msg}")
        print()

    print("=" * 78)
    print(f"Summary: {total - failed}/{total} checks passed"
          + (f" ({failed} FAIL)" if failed else "")
          + (f" — {len(missing_sections)} sections missing" if missing_sections else ""))
    print("=" * 78)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
