"""Multi-direction steering test on parity (top-k PCA baseline).

Tests whether a multi-direction linear-subspace extension of mean-difference
steering rescues regime-2 concepts that single-direction mean-difference and
single-feature SAE steering do not move. This is the empirical leg of the
"method-agnostic claim only validated on 2 methods" critique: it adds a third
linear-additive method (top-k PCA on contrastive residuals) and tests it on
parity, the genuinely-nonlinear case from §4.4.

Algorithm (top-k PCA on contrastive residuals):
  Given residuals h_target (n_t × d) and h_other (n_o × d) at the test layer,
  pool the two groups, mean-center, and take the top-k principal components
  (right singular vectors of the centered matrix). The top-k principal
  components span the largest-variance linear subspace of the contrastive set;
  if the concept lives in a low-rank linear subspace beyond the class-mean
  axis, those PCs should capture it.

  We separately compute the class-mean-difference direction d_MDS (the
  standard k=1 single-direction baseline used in §4.4) and report it
  alongside.

  This is a standard low-rank linear-subspace baseline. We do *not* claim it
  is Wollschläger et al.'s concept-cone algorithm, which uses causal-
  independence ablation tests to identify multiple behavior-relevant
  directions; their full method is more sophisticated and we leave that
  comparison to future work.

Evaluation: for each k, sum the top-k unit-norm principal components, scale
the composite to ||d_MDS|| so the comparison isolates direction (not
magnitude), and measure ΔP via the §4.4 protocol (left-padded tokenizer,
position-ids fix, injection at last real token).

Output: results/multi_direction/multi_direction_{prompts}_l{layer}_{model}.json

Usage example (parity on Gemma-2-2B at L22):
  python scripts/run_multi_direction_test.py \\
      --model google/gemma-2-2b --layer 22 --device cuda \\
      --prompts parity_prompts.json --out parity_l22_gemma2b_multidir.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO / "results/sae_validation"   # reuse SAE prompt files
RESULTS = REPO / "results/multi_direction"
RESULTS.mkdir(parents=True, exist_ok=True)


def fix_position_ids(inputs):
    inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return inputs


def get_target_token_id(tokenizer, target):
    for text in [" " + target, target]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"target {target!r} does not tokenize to a single token")


@torch.no_grad()
def extract_residuals_at_layer(model, tokenizer, prompts, layer, device,
                                batch_size=16, max_length=128, fix_pos=True):
    out_acts = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        if fix_pos:
            fix_position_ids(inputs)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        h = outputs.hidden_states[layer + 1][:, -1, :]
        out_acts.append(h.detach().cpu())
    return torch.cat(out_acts, dim=0)


@torch.no_grad()
def measure_target_p(model, tokenizer, prompts, target_tid, device,
                     steering_dir=None, alpha=1.0, layer=None,
                     batch_size=16, max_length=128):
    sd = steering_dir.to(device) if steering_dir is not None else None

    def make_hook(seq_len):
        def hook(module, input, output):
            if isinstance(output, tuple):
                resid, *rest = output
                resid = resid.clone()
                resid[:, seq_len - 1, :] = resid[:, seq_len - 1, :] + alpha * sd
                return (resid, *rest)
            else:
                output = output.clone()
                output[:, seq_len - 1, :] = output[:, seq_len - 1, :] + alpha * sd
                return output
        return hook

    probs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        fix_position_ids(inputs)
        seq_len = inputs["input_ids"].shape[1]
        handle = None
        if sd is not None:
            handle = model.model.layers[layer].register_forward_hook(make_hook(seq_len))
        try:
            outputs = model(**inputs, return_dict=True)
        finally:
            if handle is not None:
                handle.remove()
        p = F.softmax(outputs.logits[:, -1, :].float(), dim=-1)[:, target_tid]
        probs.append(p.detach().cpu())
    return torch.cat(probs).numpy()


def extract_top_k_pca_directions(target_resid, other_resid, k):
    """Top-k PCA on pooled contrastive residuals + class-mean-difference.

    Returns:
        pcs:            list of k unit-norm principal components (top-k by
                        explained variance of the pooled mean-centered residuals)
        mean_diff:      class-mean-difference vector (full, pre-normalisation)
        mean_diff_norm: ||mean_diff||  (used for scaling composite directions)
        singvals:       top-k singular values of the centered pooled matrix
        cos_pc1_mds:    cosine between top-1 PC and mean-difference direction
                        (sanity check: how aligned are the two single-direction
                        choices?)
    """
    T = target_resid.float()
    O = other_resid.float()

    # Class-mean-difference direction (the §4.4 single-direction baseline)
    mean_diff = T.mean(dim=0) - O.mean(dim=0)
    mean_diff_norm = mean_diff.norm().item()

    # Top-k PCA on pooled mean-centered residuals
    pooled = torch.cat([T, O], dim=0)
    centered = pooled - pooled.mean(dim=0, keepdim=True)
    # Truncated SVD: rows of Vh are principal components (right singular vectors)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    k_eff = min(k, Vh.shape[0])
    pcs = [Vh[i].clone() for i in range(k_eff)]

    pc1_unit = pcs[0]
    md_unit = mean_diff / (mean_diff_norm + 1e-12)
    cos_pc1_mds = float(torch.dot(pc1_unit, md_unit).abs())

    return pcs, mean_diff, mean_diff_norm, S[:k_eff].tolist(), cos_pc1_mds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompts", default="parity_prompts.json")
    ap.add_argument("--out", default="parity_l22_gemma2b_multidir.json")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10, 20],
                    help="Multi-direction k values to test")
    ap.add_argument("--max-other", type=int, default=98)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = ap.parse_args()

    device = args.device
    model_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    PROMPTS_PATH = PROMPTS_DIR / args.prompts
    OUT_PATH = RESULTS / args.out

    with open(PROMPTS_PATH) as f:
        d = json.load(f)
    target = d["target"]
    target_prompts = d["target_prompts"]
    other_prompts = d["other_prompts"][:args.max_other]
    print(f"Target {target!r}: {len(target_prompts)} target prompts, "
          f"{len(other_prompts)} contrast prompts")

    print(f"\nLoading {args.model} (dtype={args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_dtype, device_map=device,
    )
    model.eval()
    target_tid = get_target_token_id(tokenizer, target)
    print(f"Target token id: {target_tid} ({tokenizer.decode([target_tid])!r})")

    print(f"\nExtracting residuals at L{args.layer}...")
    target_resid = extract_residuals_at_layer(
        model, tokenizer, target_prompts, args.layer, device).to(device)
    other_resid = extract_residuals_at_layer(
        model, tokenizer, other_prompts, args.layer, device).to(device)
    print(f"  target residual: {target_resid.shape}, other residual: {other_resid.shape}")

    K = max(args.ks)
    print(f"\nExtracting top-{K} PCs of pooled mean-centered residuals "
          f"and class-mean-difference direction...")
    pcs, mean_diff, mean_diff_norm, singvals, cos_pc1_mds = extract_top_k_pca_directions(
        target_resid.cpu(), other_resid.cpu(), K)
    print(f"  mean-diff norm: {mean_diff_norm:.3f}")
    print(f"  top-{K} singular values: {[f'{s:.2f}' for s in singvals[:10]]}")
    print(f"  |cos(PC1, d_MDS)| = {cos_pc1_mds:.3f}  "
          f"(1.0 means PC1 ≡ mean-diff direction)")

    # Baseline P(target) on contrast prompts
    print("\nMeasuring baseline P(target)...")
    baseline_p = measure_target_p(
        model, tokenizer, other_prompts, target_tid, device,
        batch_size=args.batch_size)
    print(f"  baseline P({target!r}) = {baseline_p.mean():.5f}")

    # Single-direction MDS reference: §4.4 mean-difference baseline
    print("\nSingle-direction MDS reference:")
    sp_mds = measure_target_p(
        model, tokenizer, other_prompts, target_tid, device,
        steering_dir=mean_diff, alpha=1.0, layer=args.layer,
        batch_size=args.batch_size)
    dp_mds = float((sp_mds - baseline_p).mean())
    print(f"  k=1 (mean-difference) ΔP = {dp_mds:+.5f}")

    # Multi-direction steering: sum of top-k unit PCs, scaled to MDS norm
    print(f"\nMulti-direction PCA steering for k in {args.ks}:")
    multidir_results = {}
    for k in args.ks:
        if k > len(pcs):
            print(f"  k={k}: skipping (only {len(pcs)} PCs available)")
            continue
        composite = sum(pcs[i] for i in range(k))  # unit-norm PCs
        composite_unit = composite / (composite.norm().item() + 1e-12)
        steering_dir = composite_unit * mean_diff_norm
        sp = measure_target_p(
            model, tokenizer, other_prompts, target_tid, device,
            steering_dir=steering_dir, alpha=1.0, layer=args.layer,
            batch_size=args.batch_size)
        dp = float((sp - baseline_p).mean())
        multidir_results[str(k)] = {
            "k": k,
            "delta_p": dp,
            "steered_p_mean": float(sp.mean()),
        }
        print(f"  k={k:>2d}: ΔP = {dp:+.5f}")

    output = {
        "config": vars(args),
        "target": target,
        "target_tid": target_tid,
        "n_target": len(target_prompts),
        "n_other": len(other_prompts),
        "baseline_p_mean": float(baseline_p.mean()),
        "mean_diff_norm": mean_diff_norm,
        "top_k_singular_values": singvals,
        "cos_pc1_mds": cos_pc1_mds,
        "single_direction_mds": {"k": 1, "delta_p": dp_mds,
                                  "steered_p_mean": float(sp_mds.mean())},
        "multi_direction_pca_results": multidir_results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")

    # Headline summary
    print("\n=== Headline ===")
    print(f"baseline P({target!r}) = {output['baseline_p_mean']:.5f}")
    print(f"|cos(PC1, d_MDS)| = {cos_pc1_mds:.3f}")
    print(f"k=1 (single-direction MDS) : ΔP = {dp_mds:+.5f}")
    for k, r in multidir_results.items():
        label = f"k={k} (top-k PCA composite)"
        print(f"{label:30s}: ΔP = {r['delta_p']:+.5f}")


if __name__ == "__main__":
    main()
