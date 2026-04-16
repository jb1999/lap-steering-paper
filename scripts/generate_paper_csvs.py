"""Regenerate paper figure data CSVs from current regression results.

By default writes to ./paper_csvs/ in the repo root. Override with --out.
"""

import scripts._env  # noqa: F401

import argparse
import csv
import json
from pathlib import Path

RESULTS = Path("results")
OUT = Path("paper_csvs")


def fig1_emergence():
    """layer, family×{alin, amlp}"""
    with open(RESULTS / "exp1" / "layer_accuracy.json") as f:
        lin = json.load(f)
    with open(RESULTS / "exp1" / "mlp_probe_results.json") as f:
        mlp = json.load(f)

    families = ["analogy", "arithmetic", "geography", "sequence", "word_transform"]
    layers = sorted({int(l) for l in lin[families[0]]["layers"].keys()})

    header = ["layer"]
    for fam in families:
        tag = fam.replace("_", "")
        header += [f"{tag}alin", f"{tag}amlp"]

    rows = []
    for layer in layers:
        row = [layer]
        for fam in families:
            lin_d = lin[fam]["layers"]
            mlp_d = mlp.get(fam, {})
            la = lin_d.get(str(layer), {}).get("linear_acc", 0.0) or 0.0
            ma = mlp_d.get(str(layer), {}).get("mlp_acc", 0.0) or 0.0
            row += [round(la, 4), round(ma, 4)]
        rows.append(row)

    out = OUT / "fig1_emergence.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


def fig2_per_model(model_dir, out_name):
    """family, peakalin, maxdp -- only c_* (controlled) families."""
    p = RESULTS / "cross_concept" / model_dir / "results.json"
    if not p.exists():
        print(f"SKIP {p} (missing)")
        return
    with open(p) as f:
        data = json.load(f)

    rows = []
    families_dict = data.get("families", data)
    for fam_name, fam in sorted(families_dict.items()):
        if not fam_name.startswith("c_"):
            continue
        peak_alin = fam.get("best_alin", 0.0) or 0.0
        max_dp = fam.get("max_steering_dp")
        if max_dp is None:
            max_dp = 0.0  # placeholder for excluded families
        # Strip "c_" prefix to match original format
        rows.append([fam_name[2:], round(peak_alin, 4), round(max_dp, 5)])

    out = OUT / out_name
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "peakalin", "maxdp"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


def fig3_chaotic():
    """lambda, dp, alin -- per layer per family, concatenated."""
    with open(RESULTS / "exp1" / "layer_accuracy.json") as f:
        lin = json.load(f)
    with open(RESULTS / "exp1" / "geometric_metrics.json") as f:
        geo = json.load(f)
    with open(RESULTS / "exp2" / "steering_results.json") as f:
        steer = json.load(f)

    families = ["analogy", "arithmetic", "geography", "sequence", "word_transform"]
    rows = []
    for fam in families:
        lin_d = lin[fam]["layers"]
        geo_d = geo[fam]
        steer_d = steer[fam]["layers"]
        layers = sorted({int(l) for l in lin_d.keys()})
        for layer in layers:
            l_str = str(layer)
            la = lin_d.get(l_str, {}).get("linear_acc", 0.0) or 0.0
            lam = geo_d.get(l_str, {}).get("mean_lambda", 0.0) or 0.0
            dp = steer_d.get(l_str, {}).get("mean_delta_p", 0.0) or 0.0
            rows.append([round(lam, 1), round(dp, 5), round(la, 4)])

    out = OUT / "fig3_chaotic.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "dp", "alin"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


def fig4_refusal():
    """layer, separability, cohensd, deltarefusal, deltacomply"""
    p = RESULTS / "refusal_demo" / "refusal_results.json"
    if not p.exists():
        print(f"SKIP {p} (missing)")
        return
    with open(p) as f:
        data = json.load(f)

    sep_data = data["layer_separability"]
    steer_data = data["steering"]
    layers = sorted({int(k) for k in sep_data.keys()})
    rows = []
    for layer in layers:
        sd = sep_data[str(layer)]
        st = steer_data.get(str(layer), {})
        rows.append([
            layer,
            round(sd.get("linear_acc", 0.0), 4),
            round(sd.get("cohens_d", 0.0), 3),
            round(st.get("delta_refusal"), 5) if st.get("delta_refusal") is not None else "",
            round(st.get("delta_comply"), 5) if st.get("delta_comply") is not None else "",
        ])

    out = OUT / "fig4_refusal.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "separability", "cohensd", "deltarefusal", "deltacomply"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT), help="Output directory for CSVs")
    args = parser.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    fig1_emergence()
    fig2_per_model("gemma-2-2b", "fig2_gemma22b.csv")
    fig2_per_model("Qwen2.5-1.5B", "fig2_Qwen2515B.csv")
    fig2_per_model("Qwen2.5-7B", "fig2_Qwen257B.csv")
    fig2_per_model("Llama-3.1-8B", "fig2_Llama318B.csv")
    fig2_per_model("pythia-2.8b-deduped", "fig2_pythia28bdeduped.csv")
    fig3_chaotic()
    fig4_refusal()
    print("done.")


if __name__ == "__main__":
    main()
