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
        "refusal.rho":       (0.808, 0.04),
    },
    "probe_baseline": {
        # Probe-derived steering on 25 controlled binary families (Gemma-2-2B).
        # Headline: separability does not predict steerability even with the
        # probe's full weight vector. See App. F.
        "n_families":               (25,     0),
        "mean_probe_acc":           (1.000,  0.01),
        "pooled.rho_acc_dpprobe":   (-0.06,  0.10),
        "pooled.rho_mds_dpprobe":   (+0.96,  0.05),
        "median.max_dp_probe":      (0.016,  0.01),
        "median.max_dp_mds":        (0.019,  0.01),
        "median.max_dp_random":     (0.0006, 0.002),
    },
    "multi_direction": {
        # Top-k PCA composite on Gemma-2-2B at L22, scaled to mean-difference
        # norm. Headline: variance-aligned multi-direction sums under-perform
        # single-direction MDS on both targets. See §4.4.
        "geography.max_dp_pca":     (0.015,  0.015),
        "geography.dp_mds":         (0.240,  0.05),
        "parity.max_dp_pca":        (0.017,  0.015),
        "parity.dp_mds":            (0.018,  0.015),
    },
    "sae_validation": {
        # SAE feature steering across three releases (App. G). For each
        # model x target: rho(C(v_f), ΔP), rho(activation, ΔP), and the peak
        # multi-feature ΔP by C(v_f) ranking; MDS baseline at the same layer.
        # Gemma-2-2B (L22), GemmaScope, geography → "Spanish":
        "gemma2b.geo.rho_cv":       (+0.629, 0.10),
        "gemma2b.geo.rho_act":      (-0.158, 0.15),
        "gemma2b.geo.dp_top20_cv":  (+0.324, 0.05),
        "gemma2b.geo.dp_mds":       (+0.240, 0.05),
        # Gemma-2-2B (L22), parity → "odd":
        "gemma2b.par.rho_cv":       (+0.754, 0.10),
        "gemma2b.par.rho_act":      (-0.462, 0.15),
        # Gemma-2-9B (L36), GemmaScope:
        "gemma9b.geo.rho_cv":       (+0.632, 0.10),
        "gemma9b.geo.rho_act":      (-0.220, 0.15),
        "gemma9b.geo.dp_top20_cv":  (+0.286, 0.05),
        "gemma9b.geo.dp_mds":       (+0.208, 0.05),
        "gemma9b.par.rho_cv":       (+0.466, 0.12),
        "gemma9b.par.rho_act":      (-0.456, 0.15),
        # Llama-3.1-8B (L27), Llama-Scope:
        "llama8b.geo.rho_cv":       (+0.819, 0.08),
        "llama8b.geo.rho_act":      (-0.415, 0.15),
        "llama8b.geo.dp_top5_cv":   (+0.444, 0.08),
        "llama8b.geo.dp_mds":       (+0.324, 0.06),
        "llama8b.par.rho_cv":       (+0.658, 0.10),
        "llama8b.par.rho_act":      (-0.458, 0.15),
    },
    "arad_h2h": {
        # Head-to-head with Arad et al. (2025) on 4 (model, layer) panels x
        # 2 targets = 8 cells (App. arad_h2h). Two headline correlations per
        # panel: rho(C_t(v), DeltaP_ours) shows our static score predicts our
        # intervention regime; rho(S_out^target, DeltaP_arad) reproduces their
        # target-conditioned score predicting their amplification regime.
        # Deterministic (seed=42, no probe training) so tolerances are tight.
        "geo_l22_gemma2b.rho_cv_ours":      (+0.480, 0.05),
        "geo_l22_gemma2b.rho_soutgt_arad":  (+0.920, 0.04),
        "geo_l27_llama8b.rho_cv_ours":      (+0.536, 0.05),
        "geo_l27_llama8b.rho_soutgt_arad":  (+0.994, 0.02),
        "geo_l31_gemma9bit.rho_cv_ours":    (+0.336, 0.05),
        "geo_l31_gemma9bit.rho_soutgt_arad":(+0.958, 0.04),
        "geo_l36_gemma9b.rho_cv_ours":      (+0.418, 0.05),
        "geo_l36_gemma9b.rho_soutgt_arad":  (+0.863, 0.05),
        "par_l22_gemma2b.rho_cv_ours":      (+0.539, 0.05),
        "par_l22_gemma2b.rho_soutgt_arad":  (+0.922, 0.04),
        "par_l27_llama8b.rho_cv_ours":      (+0.436, 0.05),
        "par_l27_llama8b.rho_soutgt_arad":  (+0.971, 0.03),
        "par_l31_gemma9bit.rho_cv_ours":    (+0.352, 0.05),
        "par_l31_gemma9bit.rho_soutgt_arad":(+0.907, 0.04),
        "par_l36_gemma9b.rho_cv_ours":      (+0.540, 0.05),
        "par_l36_gemma9b.rho_soutgt_arad":  (+0.876, 0.04),
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


def compute_probe_baseline():
    """Probe-derived steering on 25 controlled binary families (App. F)."""
    import math
    import numpy as np
    p = R / "probe_steering" / "probe_baseline_gemma-2-2b_controlled.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    fams = d.get("families", {})
    if not fams:
        return None
    out = {"n_families": len(fams)}
    accs, mds_max, probe_max, rand_max = [], [], [], []
    pooled_acc, pooled_dp_probe, pooled_dp_mds = [], [], []
    for fam, fd in fams.items():
        s = fd.get("summary", {})
        if isinstance(s.get("mean_probe_acc"), (int, float)):
            accs.append(s["mean_probe_acc"])
        for src, dst in [("max_dp_mds", mds_max), ("max_dp_probe", probe_max),
                         ("max_dp_random", rand_max)]:
            v = s.get(src)
            if isinstance(v, (int, float)) and math.isfinite(v):
                dst.append(v)
        for row in fd.get("rows", []):
            acc = row.get("probe_train_acc")
            dpp = row.get("delta_p_probe")
            dpm = row.get("delta_p_mds")
            if all(isinstance(x, (int, float)) and math.isfinite(x)
                   for x in (acc, dpp, dpm)):
                pooled_acc.append(acc)
                pooled_dp_probe.append(dpp)
                pooled_dp_mds.append(dpm)
    if accs:
        out["mean_probe_acc"] = float(np.mean(accs))
    if mds_max:
        out["median.max_dp_mds"] = float(np.median(mds_max))
    if probe_max:
        out["median.max_dp_probe"] = float(np.median(probe_max))
    if rand_max:
        out["median.max_dp_random"] = float(np.median(rand_max))
    if pooled_acc:
        out["pooled.rho_acc_dpprobe"] = spearmanr(pooled_acc, pooled_dp_probe)[0]
        out["pooled.rho_mds_dpprobe"] = spearmanr(pooled_dp_mds, pooled_dp_probe)[0]
    return out


def compute_multi_direction():
    """Top-k PCA composite vs MDS on Gemma-2-2B L22 (§4.4)."""
    out = {}
    for fname, prefix in [
        ("geography_l22_gemma2b_multidir.json", "geography"),
        ("parity_l22_gemma2b_multidir.json",   "parity"),
    ]:
        p = R / "multi_direction" / fname
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        # Defensive: target may identify which family ran.
        target = d.get("target", "").lower()
        # Map: target token "Spanish" → geography; "odd" → parity.
        actual_prefix = "geography" if "spanish" in target else (
                        "parity"    if "odd"     in target else prefix)
        results = d.get("multi_direction_pca_results")
        if isinstance(results, dict):
            dps = [r.get("delta_p") for r in results.values()
                   if isinstance(r.get("delta_p"), (int, float))]
            if dps:
                out[f"{actual_prefix}.max_dp_pca"] = max(dps)
        mds = d.get("single_direction_mds", {}).get("delta_p")
        if isinstance(mds, (int, float)):
            out[f"{actual_prefix}.dp_mds"] = mds
    return out


def compute_sae_validation():
    """SAE feature steering across three releases (App. G)."""
    out = {}
    cases = [
        ("gemma2b.geo", "sae_geography_l22_gemma2b.json", 20),
        ("gemma2b.par", "sae_parity_l22_gemma2b.json",    20),
        ("gemma9b.geo", "sae_geography_l36_gemma9b.json", 20),
        ("gemma9b.par", "sae_parity_l36_gemma9b.json",    20),
        ("llama8b.geo", "sae_geography_l27_llama8b.json", 5),
        ("llama8b.par", "sae_parity_l27_llama8b.json",    100),
    ]
    for tag, fname, k_top in cases:
        p = R / "sae_validation" / fname
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        corr = d.get("correlations", {})
        cv_block = corr.get("C_v_target_vs_dp", {})
        act_block = corr.get("activation_vs_dp", {})
        if isinstance(cv_block.get("rho"), (int, float)):
            out[f"{tag}.rho_cv"] = cv_block["rho"]
        if isinstance(act_block.get("rho"), (int, float)):
            out[f"{tag}.rho_act"] = act_block["rho"]
        mds = d.get("mean_diff", {}).get("delta_p")
        if isinstance(mds, (int, float)):
            out[f"{tag}.dp_mds"] = mds
        # Multi-feature ΔP at the chosen k by C(v_f) ranking. Keys are "k=1", "k=5", ...
        mf = d.get("multi_feature_by_C_v_target", {})
        entry = mf.get(f"k={k_top}")
        if isinstance(entry, dict) and isinstance(entry.get("delta_p"), (int, float)):
            label = "dp_top5_cv" if k_top == 5 else "dp_top20_cv"
            out[f"{tag}.{label}"] = entry["delta_p"]
    return out


def compute_arad_h2h():
    """Head-to-head with Arad et al. (2025) on 8 panels (App. arad_h2h)."""
    out = {}
    panels = [
        ("geo_l22_gemma2b",  "arad_h2h_geography_l22_gemma2b.json"),
        ("geo_l27_llama8b",  "arad_h2h_geography_l27_llama8b.json"),
        ("geo_l31_gemma9bit","arad_h2h_geography_l31_gemma9bit.json"),
        ("geo_l36_gemma9b",  "arad_h2h_geography_l36_gemma9b.json"),
        ("par_l22_gemma2b",  "arad_h2h_parity_l22_gemma2b.json"),
        ("par_l27_llama8b",  "arad_h2h_parity_l27_llama8b.json"),
        ("par_l31_gemma9bit","arad_h2h_parity_l31_gemma9bit.json"),
        ("par_l36_gemma9b",  "arad_h2h_parity_l36_gemma9b.json"),
    ]
    for tag, fname in panels:
        p = R / "sae_validation" / fname
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        feats = d.get("feature_results", [])
        if not feats:
            continue
        cv = [r["C_v_target"] for r in feats]
        sout_tgt = [r["S_out_target"] for r in feats]
        dp_ours = [r["delta_p_ours"] for r in feats]
        dp_arad = [r.get("delta_p_arad") for r in feats]
        rho_cv_ours, _ = spearmanr(cv, dp_ours)
        out[f"{tag}.rho_cv_ours"] = float(rho_cv_ours)
        if all(x is not None for x in dp_arad):
            rho_soutgt_arad, _ = spearmanr(sout_tgt, dp_arad)
            out[f"{tag}.rho_soutgt_arad"] = float(rho_soutgt_arad)
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
    ("PROBE BASELINE — separability != steerability (n=25, App. F)",
     "probe_baseline", compute_probe_baseline),
    ("MULTI-DIRECTION — top-k PCA control (Gemma-2-2B, §4.4)",
     "multi_direction", compute_multi_direction),
    ("SAE VALIDATION — output-alignment vs activation (App. G)",
     "sae_validation", compute_sae_validation),
    ("ARAD H2H — C_t(v) vs target-conditioned S_out (App. arad_h2h)",
     "arad_h2h", compute_arad_h2h),
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
