"""Compute all statistics needed for paper tables from current results."""

import scripts._env  # noqa: F401

import json
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np

R = Path("results")

print("=" * 70)
print("EXP1 / EXP2 -- Per-family on Gemma-2-2B (Tables 2 and 3)")
print("=" * 70)

with open(R / "exp1" / "layer_accuracy.json") as f:
    lin = json.load(f)
with open(R / "exp1" / "mlp_probe_results.json") as f:
    mlp = json.load(f)
with open(R / "exp2" / "steering_results.json") as f:
    steer = json.load(f)
with open(R / "exp1" / "geometric_metrics.json") as f:
    geo = json.load(f)

families = ["arithmetic", "geography", "sequence", "word_transform", "analogy"]

print(f"{'family':<16} {'acc':>6} {'best_alin':>10} {'L':>3} {'best_amlp':>10} {'L':>3} {'gap':>6} {'maxDP':>8} {'L':>3}")
for fam in families:
    acc = lin[fam]["model_top1"]
    lin_layers = {int(l): lin[fam]["layers"][l]["linear_acc"] for l in lin[fam]["layers"]}
    mlp_layers = {int(l): mlp[fam][l]["mlp_acc"] for l in mlp[fam]}
    steer_layers = {int(l): steer[fam]["layers"][l]["mean_delta_p"] for l in steer[fam]["layers"]}
    best_alin_layer = max(lin_layers, key=lin_layers.get)
    best_alin = lin_layers[best_alin_layer]
    best_amlp_layer = max(mlp_layers, key=mlp_layers.get)
    best_amlp = mlp_layers[best_amlp_layer]
    best_dp_layer = max(steer_layers, key=steer_layers.get)
    best_dp = steer_layers[best_dp_layer]
    gap = best_amlp - best_alin
    print(f"{fam:<16} {acc:>6.3f} {best_alin:>10.4f} {best_alin_layer:>3} {best_amlp:>10.4f} {best_amlp_layer:>3} {gap:>6.3f} {best_dp:>8.4f} {best_dp_layer:>3}")

print()
print("=" * 70)
print("EXP2 -- Per-family steering correlations on Gemma (Table 3)")
print("=" * 70)

# rho(alin, dp) and rho(lambda, dp) per family across layers
all_alin, all_dp, all_lam = [], [], []
for fam in families:
    lin_layers = {int(l): lin[fam]["layers"][l]["linear_acc"] for l in lin[fam]["layers"]}
    steer_layers = {int(l): steer[fam]["layers"][l]["mean_delta_p"] for l in steer[fam]["layers"]}
    geo_layers = {int(l): geo[fam][l]["mean_lambda"] for l in geo[fam]}

    layers = sorted(set(lin_layers.keys()) & set(steer_layers.keys()))
    a = [lin_layers[l] for l in layers]
    d = [steer_layers[l] for l in layers]
    g = [geo_layers[l] for l in layers]

    rho_ad, p_ad = spearmanr(a, d)
    rho_ld, p_ld = spearmanr(g, d)
    print(f"{fam:<16} rho(alin,dp)={rho_ad:+.3f} (p={p_ad:.4f})  rho(lambda,dp)={rho_ld:+.3f} (p={p_ld:.4f})")

    all_alin.extend(a)
    all_dp.extend(d)
    all_lam.extend(g)

rho_pool, p_pool = spearmanr(all_alin, all_dp)
rho_lp, p_lp = spearmanr(all_lam, all_dp)
print(f"\nPOOLED (n={len(all_alin)}): rho(alin,dp)={rho_pool:+.3f} (p={p_pool:.2e})  rho(lambda,dp)={rho_lp:+.3f} (p={p_lp:.2e})")

print()
print("=" * 70)
print("CROSS-CONCEPT TABLE (tab:crossconcept) -- across 5 models")
print("=" * 70)

cc_models = {
    "Pythia-2.8B": "pythia-2.8b-deduped",
    "Gemma-2-2B": "gemma-2-2b",
    "Qwen-1.5B": "Qwen2.5-1.5B",
    "Qwen-7B": "Qwen2.5-7B",
    "Llama-8B": "Llama-3.1-8B",
}

for nice, dirname in cc_models.items():
    p = R / "cross_concept" / dirname / "results.json"
    if not p.exists():
        print(f"{nice}: MISSING")
        continue
    with open(p) as f:
        d = json.load(f)
    fams = d["families"]

    # All families with steering data
    all_pairs = [(f["best_alin"], f["max_steering_dp"]) for f in fams.values() if f.get("max_steering_dp") is not None]
    all_alin = [a for a, _ in all_pairs]
    all_dp = [d for _, d in all_pairs]
    rho_all, _ = spearmanr(all_alin, all_dp)

    # Controlled (c_*) only
    ctrl_pairs = [(f["best_alin"], f["max_steering_dp"]) for n, f in fams.items() if n.startswith("c_") and f.get("max_steering_dp") is not None]
    ctrl_alin = [a for a, _ in ctrl_pairs]
    ctrl_dp = [d for _, d in ctrl_pairs]
    rho_ctrl, _ = spearmanr(ctrl_alin, ctrl_dp)

    # Above-floor: ctrl with alin > 0.05
    floor05 = [(a, d) for a, d in ctrl_pairs if a > 0.05]
    rho_05 = spearmanr([x[0] for x in floor05], [x[1] for x in floor05])[0] if len(floor05) >= 3 else float('nan')

    # Above-floor: ctrl with alin > 0.1
    floor10 = [(a, d) for a, d in ctrl_pairs if a > 0.1]
    rho_10 = spearmanr([x[0] for x in floor10], [x[1] for x in floor10])[0] if len(floor10) >= 3 else float('nan')

    print(f"{nice:<14} all: {rho_all:+.2f} (n={len(all_pairs)})  ctrl: {rho_ctrl:+.2f} (n={len(ctrl_pairs)})  >.05: {rho_05:+.2f} (n={len(floor05)})  >.10: {rho_10:+.2f} (n={len(floor10)})")

print()
print("=" * 70)
print("SCALING TABLE (tab:scaling) -- Pythia 160M to 6.9B")
print("=" * 70)

with open(R / "scaling" / "scaling_results.json") as f:
    scaling = json.load(f)
print(json.dumps(scaling, indent=2))

print()
print("=" * 70)
print("REPLICATION (Table 4) -- per-family peak alin and max dp")
print("=" * 70)

for nice, dirname in [
    ("Llama-3.1-8B", "replication_Llama-3.1-8B"),
    ("Mistral-7B", "replication_Mistral-7B-v0.3"),
    ("Qwen2.5-7B", "replication_Qwen2.5-7B"),
]:
    p = R / dirname / "results.json"
    if not p.exists():
        print(f"{nice}: MISSING")
        continue
    with open(p) as f:
        d = json.load(f)
    print(f"\n{nice}:")
    for fam in families:
        if fam not in d:
            continue
        fd = d[fam]
        acc = fd.get("model_accuracy", "?")
        la = fd.get("layer_accuracy", {})
        if la:
            best_l = max(la, key=lambda k: la[k])
            best_a = la[best_l]
        else:
            best_l, best_a = "?", "?"
        st = fd.get("steering", {})
        if st:
            best_sl = max(st, key=lambda k: st[k])
            best_sd = st[best_sl]
        else:
            best_sl, best_sd = "?", "?"
        print(f"  {fam:<14} acc={acc} best_alin={best_a:.3f} (L{best_l}) max_dp={best_sd if best_sd == '?' else f'{best_sd:.3f}'} (L{best_sl})")

print()
print("=" * 70)
print("ENTITY STEERING + REFUSAL")
print("=" * 70)

for nice, p in [
    ("Gemma entity", R / "entity_steering_demo" / "entity_steering_results.json"),
    ("OLMo entity", R / "entity_steering_olmo" / "entity_steering_demo" / "entity_steering_results.json"),
    ("Refusal", R / "refusal_demo" / "refusal_results.json"),
]:
    if not p.exists():
        print(f"{nice}: MISSING")
        continue
    with open(p) as f:
        d = json.load(f)
    print(f"\n{nice}:")
    if "correlation_rho" in d:
        print(f"  correlation_rho = {d['correlation_rho']:.3f} (p={d.get('correlation_p', '?')})")
    if "layer_separability" in d:
        seps = {int(k): v.get("linear_acc", 0) for k, v in d["layer_separability"].items()}
        peak_l = max(seps, key=seps.get)
        print(f"  peak separability: {seps[peak_l]:.3f} at L{peak_l}")
    if "best_alin_layer" in d:
        print(f"  best_alin_layer = L{d['best_alin_layer']}, best_alin = {d.get('best_alin', '?')}")
