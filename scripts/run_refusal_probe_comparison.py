"""
Follow-up to run_refusal_demo.py: compare per-layer probe accuracy under
three readouts on the same 80 prompts.

  raw    : LR probe trained on h_l directly (the existing §4.3 method)
  ln     : LR probe trained on LN_final(h_l)
  logit  : LR probe trained on W_U @ LN_final(h_l)  (full vocab logits)

Reports the three accuracy curves and their Spearman correlation with the
steering DeltaP curve already saved in results/refusal_demo/refusal_results.json.

Layer selection should be robust to the readout choice if the §4.3 probe is
serving as a layer-selection heuristic; if the curves disagree on which layer
peaks we have a methodological issue worth flagging.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.extraction.activations import ActivationExtractor


HARMFUL_PROMPTS_FILE = Path(__file__).resolve().parent / "run_refusal_demo.py"


def load_prompts():
    """Reuse the prompt lists in run_refusal_demo.py without importing it."""
    src = HARMFUL_PROMPTS_FILE.read_text()
    ns = {}
    # extract the two list literals
    start_h = src.index("HARMFUL_PROMPTS = [")
    end_h = src.index("]", start_h) + 1
    start_b = src.index("BENIGN_PROMPTS = [")
    end_b = src.index("]", start_b) + 1
    exec(src[start_h:end_h], ns)
    exec(src[start_b:end_b], ns)
    return ns["HARMFUL_PROMPTS"], ns["BENIGN_PROMPTS"]


def lr_cv_acc(features: np.ndarray, labels: np.ndarray, seed: int = 42) -> float:
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    lr = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
    return float(cross_val_score(lr, X, labels, cv=5, scoring="accuracy").mean())


def build_readout(extractor):
    """Return (ln_module, lm_head_module) from a HF causal LM extractor."""
    model = extractor.model
    if hasattr(model.model, "norm"):
        ln = model.model.norm
    elif hasattr(model.model, "final_layernorm"):
        ln = model.model.final_layernorm
    else:
        raise RuntimeError("could not locate final layer norm")
    return ln, model.lm_head


@torch.no_grad()
def project(activations_l: np.ndarray, ln, lm_head, mode: str, device: str) -> np.ndarray:
    """activations_l: (n_prompts, d).  Returns (n_prompts, k) for the chosen readout."""
    h = torch.from_numpy(activations_l).to(device=device, dtype=next(ln.parameters()).dtype)
    if mode == "raw":
        return activations_l
    if mode == "ln":
        out = ln(h)
        return out.float().cpu().numpy()
    if mode == "logit":
        out = lm_head(ln(h))
        return out.float().cpu().numpy()
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument(
        "--readout",
        nargs="+",
        default=["raw", "ln", "logit"],
        choices=["raw", "ln", "logit"],
        help="which readout(s) to compute. raw = §4.3 default, ln = post-final-LN, logit = post-LN+W_U",
    )
    ap.add_argument("--inference-batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir) / "refusal_demo"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else results_dir / "probe_readout_comparison.json"

    harmful, benign = load_prompts()
    all_prompts = harmful + benign
    labels = np.array([1] * len(harmful) + [0] * len(benign))

    print(f"Model: {args.model}  |  prompts: {len(harmful)} harmful + {len(benign)} benign")
    print(f"Readouts: {args.readout}")

    extractor = ActivationExtractor(args.model, device=args.device)
    n_layers = extractor.n_layers
    ln, lm_head = build_readout(extractor)

    print(f"Extracting activations across {n_layers} layers...")
    extraction = extractor.extract(
        all_prompts, layers=list(range(n_layers)),
        batch_size=args.inference_batch_size,
    )
    activations = extraction.to_numpy()  # dict: layer -> (n, d)

    per_layer = {mode: [] for mode in args.readout}
    for l in range(n_layers):
        H = activations[l]
        line = f"  L{l:>2}:"
        for mode in args.readout:
            X = project(H, ln, lm_head, mode, args.device)
            acc = lr_cv_acc(X, labels)
            per_layer[mode].append(acc)
            line += f"  {mode}={acc:.3f}"
        print(line)

    # Correlate each curve with the steering DeltaP curve from the existing demo.
    ref_path = results_dir / "refusal_results.json"
    correlations = {}
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        steering = ref.get("steering", {})
        # steering is keyed by layer index (string); pull DeltaP_comply or similar
        layer_keys = sorted(steering.keys(), key=int)
        if layer_keys:
            example = steering[layer_keys[0]]
            dp_field = next(
                (k for k in ("delta_p_comply", "delta_p", "comply_delta", "delta_p_compliance")
                 if k in example),
                None,
            )
            if dp_field:
                steered_layers = [int(k) for k in layer_keys]
                dp_vals = [steering[str(l)][dp_field] for l in steered_layers]
                for mode in args.readout:
                    probe_vals = [per_layer[mode][l] for l in steered_layers]
                    rho, p = spearmanr(probe_vals, dp_vals)
                    correlations[mode] = {"rho": float(rho), "p": float(p), "n": len(dp_vals), "field": dp_field}
                    print(f"  rho({mode}, {dp_field}) = {rho:+.3f}  (p={p:.3g}, n={len(dp_vals)})")
            else:
                print(f"  (could not find delta-P field in steering entries; skipping correlations)")

    # Peak layer per readout
    peaks = {mode: int(np.argmax(per_layer[mode])) for mode in args.readout}
    print(f"  Peak layer per readout: {peaks}")

    out = {
        "model": args.model,
        "n_harmful": len(harmful),
        "n_benign": len(benign),
        "n_layers": n_layers,
        "per_layer_accuracy": per_layer,
        "peak_layer": peaks,
        "correlations_with_steering": correlations,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
