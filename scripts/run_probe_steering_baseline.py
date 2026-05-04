"""Probe-derived-steering baseline: tests "separability != steerability" directly.

The paper's central conceptual point is that trained probes succeed at every
layer (>93% accuracy on Gemma-2-2B for the 5 core families) while steering
fails at most of those layers. This script tests that claim quantitatively by
using the trained logistic regression probe's *weight vector* as a steering
direction and comparing the resulting steering effect to mean-difference
steering at the same layer.

Protocol mirrors scripts/run_steerability.py (the canonical paper protocol):
  - Model in bfloat16, left-padded tokenizer
  - Position-ids fix at measurement time
  - Injection at the rightmost real token (seq_len - 1)
  - Direction scaled to mean-difference norm so we compare *direction*, not magnitude
  - alpha = 1.0 (full direction injected)

For each of the 5 core families and each layer:
  1. Pick the most common correct answer as the steering target.
  2. Build target_prompts (whose correct answer == target) and other_prompts.
  3. Extract residuals at the layer for both groups.
  4. Compute three steering directions, all scaled to ||d_MDS||:
       (a) mean-difference (MDS):  d = mean(target) - mean(other)
       (b) probe weight:            w = LogisticRegressionCV(target vs other).coef_
       (c) random unit vector control
  5. Measure P(target_tid) on other_prompts: baseline + steered (each direction).
  6. Record probe train accuracy + per-layer ΔP for each direction.

Output: results/probe_steering/probe_baseline_gemma-2-2b.json

Usage:
  python scripts/run_probe_steering_baseline.py --device cuda
"""

import scripts._env  # noqa: F401

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results/probe_steering"
RESULTS.mkdir(parents=True, exist_ok=True)


def fix_position_ids(inputs):
    inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return inputs


def get_token_id(tokenizer, target):
    for text in [" " + target, target]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"target {target!r} not single-token")


@torch.no_grad()
def extract_all_layers(model, tokenizer, prompts, device, n_layers,
                       batch_size=16, max_length=128):
    """Return tensor (n_layers, n_prompts, d_model) of residuals at last position."""
    out = [[] for _ in range(n_layers)]
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(device)
        fix_position_ids(inputs)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        for l in range(n_layers):
            h = outputs.hidden_states[l + 1][:, -1, :]  # post-block residual at last pos
            out[l].append(h.detach().cpu().float())
    return [torch.cat(layer, dim=0) for layer in out]


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


def train_probe_get_weight(target_resid, other_resid, seed=42):
    """Train LR probe (target=1, other=0); return probe accuracy + weight vector.

    Balances the training set by subsampling 'other' to match 'target' size to
    avoid trivial accuracy from class imbalance (target/other ratio can be 4/496).
    """
    rng = np.random.default_rng(seed)
    n = min(len(target_resid), len(other_resid))
    if n < 5:
        return float("nan"), np.zeros(target_resid.shape[1], dtype=np.float32)
    if len(other_resid) > n:
        idx = rng.choice(len(other_resid), size=n, replace=False)
        other_bal = other_resid[idx]
    else:
        other_bal = other_resid
    X = np.concatenate([target_resid[:n], other_bal], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegressionCV(Cs=10, cv=min(5, n), penalty="l2", solver="lbfgs",
                                max_iter=2000, random_state=seed)
    clf.fit(Xs, y)
    acc = clf.score(Xs, y)
    W = clf.coef_[0] / scaler.scale_
    return acc, W


def select_target(family_name, family_prompts):
    """Match scripts/run_experiment2.py:select_steering_targets logic."""
    answer_counts = Counter(p.correct_answer for p in family_prompts)
    if family_name == "arithmetic":
        best_target = None
        best_contam = float("inf")
        for answer, count in answer_counts.most_common():
            if count < 20:
                continue
            n_contam = sum(1 for p in family_prompts
                           if p.correct_answer != answer and answer in p.prompt_text)
            if n_contam < best_contam:
                best_contam = n_contam
                best_target = answer
        return best_target or answer_counts.most_common(1)[0][0]
    for answer, count in answer_counts.most_common():
        if count >= 20 and len(family_prompts) - count >= 50:
            return answer
    return answer_counts.most_common(1)[0][0]


def run_family(model, tokenizer, family_name, family_prompts, device, n_layers,
               args, target_token_override=None):
    if target_token_override is not None:
        target = target_token_override
    elif family_name in args.family_targets:
        target = args.family_targets[family_name]
    else:
        target = select_target(family_name, family_prompts)
    target_tid = get_token_id(tokenizer, target)
    print(f"\n=== {family_name} → target {target!r} (tid {target_tid}) ===")

    target_prompts = [p for p in family_prompts if p.correct_answer == target]
    if family_name == "arithmetic":
        other_prompts = [p for p in family_prompts
                         if p.correct_answer != target and target not in p.prompt_text]
    else:
        other_prompts = [p for p in family_prompts if p.correct_answer != target]
    if len(target_prompts) < args.min_target or len(other_prompts) < 10:
        print(f"  insufficient prompts: target={len(target_prompts)} (need {args.min_target}), "
              f"other={len(other_prompts)}")
        return None

    target_texts = [p.prompt_text for p in target_prompts]
    other_texts = [p.prompt_text for p in other_prompts][:args.max_other]
    print(f"  target={len(target_texts)}, other={len(other_texts)}")

    # Extract residuals at all layers
    print("  extracting residuals...")
    target_resid = extract_all_layers(model, tokenizer, target_texts, device, n_layers,
                                       batch_size=args.batch_size)
    other_resid = extract_all_layers(model, tokenizer, other_texts, device, n_layers,
                                      batch_size=args.batch_size)

    # Baseline P(target_tid) on other prompts (no steering)
    print("  measuring baseline P(target)...")
    baseline_p = measure_target_p(model, tokenizer, other_texts, target_tid, device,
                                  batch_size=args.batch_size)
    print(f"  baseline P({target!r}) = {baseline_p.mean():.5f}")

    # Per-layer steering experiments
    rng = np.random.default_rng(args.seed)
    rows = []
    for L in range(n_layers):
        T = target_resid[L].numpy()
        O = other_resid[L].numpy()
        d_mds = T.mean(axis=0) - O.mean(axis=0)
        d_mds_norm = float(np.linalg.norm(d_mds))

        probe_acc, w_probe = train_probe_get_weight(T, O, seed=args.seed)
        w_norm = float(np.linalg.norm(w_probe)) + 1e-12
        # scale probe direction to MDS norm so we compare direction, not magnitude
        d_probe = (w_probe / w_norm) * d_mds_norm

        # random unit direction scaled to MDS norm (matched control)
        d_rand = rng.standard_normal(d_mds.shape[0]).astype(np.float32)
        d_rand = d_rand / np.linalg.norm(d_rand) * d_mds_norm

        cos_probe_mds = float(np.dot(w_probe / w_norm, d_mds / (d_mds_norm + 1e-12)))

        # Measure ΔP for each direction
        sp_mds = measure_target_p(model, tokenizer, other_texts, target_tid, device,
                                   steering_dir=torch.tensor(d_mds, dtype=torch.float32),
                                   alpha=1.0, layer=L, batch_size=args.batch_size)
        sp_probe = measure_target_p(model, tokenizer, other_texts, target_tid, device,
                                     steering_dir=torch.tensor(d_probe, dtype=torch.float32),
                                     alpha=1.0, layer=L, batch_size=args.batch_size)
        sp_rand = measure_target_p(model, tokenizer, other_texts, target_tid, device,
                                    steering_dir=torch.tensor(d_rand, dtype=torch.float32),
                                    alpha=1.0, layer=L, batch_size=args.batch_size)

        dp_mds = float((sp_mds - baseline_p).mean())
        dp_probe = float((sp_probe - baseline_p).mean())
        dp_rand = float((sp_rand - baseline_p).mean())

        rows.append({
            "layer": L,
            "probe_train_acc": float(probe_acc),
            "d_mds_norm": d_mds_norm,
            "cos_probe_mds": cos_probe_mds,
            "delta_p_mds": dp_mds,
            "delta_p_probe": dp_probe,
            "delta_p_random": dp_rand,
        })
        print(f"  L{L:>2}: probe_acc={probe_acc:.3f}  cos(w,d_MDS)={cos_probe_mds:+.3f}  "
              f"ΔP_MDS={dp_mds:+.4f}  ΔP_probe={dp_probe:+.4f}  ΔP_rand={dp_rand:+.4f}")

    # Cross-layer correlations
    arr = np.array([(r["delta_p_mds"], r["delta_p_probe"], r["delta_p_random"],
                     r["probe_train_acc"]) for r in rows])
    rho_probe_mds, _ = spearmanr(arr[:, 0], arr[:, 1])
    rho_acc_dp_probe, _ = spearmanr(arr[:, 3], arr[:, 1])
    rho_acc_dp_mds, _ = spearmanr(arr[:, 3], arr[:, 0])

    return {
        "family": family_name,
        "target": target,
        "target_tid": target_tid,
        "n_target": len(target_prompts),
        "n_other": len(other_texts),
        "baseline_p_mean": float(baseline_p.mean()),
        "rows": rows,
        "summary": {
            "rho(delta_p_mds, delta_p_probe)": float(rho_probe_mds),
            "rho(probe_acc, delta_p_probe)": float(rho_acc_dp_probe),
            "rho(probe_acc, delta_p_mds)": float(rho_acc_dp_mds),
            "max_dp_probe": float(arr[:, 1].max()),
            "max_dp_mds": float(arr[:, 0].max()),
            "max_dp_random": float(arr[:, 2].max()),
            "mean_probe_acc": float(arr[:, 3].mean()),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--families", nargs="+",
                    default=["arithmetic", "geography", "sequence", "word_transform", "analogy"])
    ap.add_argument("--n-prompts", type=int, default=500)
    ap.add_argument("--max-other", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-target", type=int, default=5,
                    help="Minimum target-prompt count to attempt the family. "
                         "Lower for sparse-answer families like word_transform / analogy.")
    ap.add_argument("--target", action="append", default=[],
                    help="Override target answer for a family. Format: family:answer. "
                         "Can be passed multiple times. Example: --target word_transform:cold")
    ap.add_argument("--out", default="probe_baseline_gemma-2-2b.json")
    args = ap.parse_args()
    args.family_targets = dict(t.split(":", 1) for t in args.target)

    from src.data.loader import load_families

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Model: {args.model}, n_layers={n_layers}")

    needs_controlled = any(f.startswith("c_") for f in args.families)
    families = load_families(n_prompts=args.n_prompts,
                             include_controlled=needs_controlled)
    print(f"Loaded {len(families)} families; running {args.families}")

    out = {"config": vars(args), "families": {}}
    for fam in args.families:
        if fam not in families:
            print(f"WARNING: family {fam!r} not found in probe_loader")
            continue
        result = run_family(model, tokenizer, fam, families[fam], args.device,
                            n_layers, args)
        if result is not None:
            out["families"][fam] = result
            # Write incrementally
            with open(RESULTS / args.out, "w") as f:
                json.dump(out, f, indent=2)

    print(f"\nDone. Results: {RESULTS / args.out}")
    # Headline summary
    print("\n=== Headline ===")
    for fam, r in out["families"].items():
        s = r["summary"]
        print(f"{fam:14s} max ΔP: MDS={s['max_dp_mds']:+.4f}  probe={s['max_dp_probe']:+.4f}  "
              f"rand={s['max_dp_random']:+.4f}  | mean probe acc={s['mean_probe_acc']:.3f}")


if __name__ == "__main__":
    main()
