"""Experiment 2: LAP Predicts Tool Success (H2)

Tests whether Exp1's LAP metrics predict where steering vectors succeed.

For each concept family:
  1. Pick a target answer token (e.g., "7" for arithmetic)
  2. Compute steering direction at each layer: mean(target) - mean(other)
  3. Inject at each layer, measure effect size and collateral
  4. Correlate with Exp1 LAP metrics (linear acc, probe gap, λ)

Usage:
    python scripts/run_experiment2.py --model google/gemma-2-2b --device cuda
"""

import scripts._env  # noqa: F401 — must be first

import argparse
import gc
import json
import torch
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from src.extraction.activations import ActivationExtractor
from src.data.loader import load_families


def select_steering_targets(families):
    """For each family, pick the most common correct answer as the steering target.

    Returns dict: family_name -> target_answer_string
    """
    from collections import Counter
    targets = {}
    for family_name, prompts in families.items():
        answer_counts = Counter(p.correct_answer for p in prompts)

        # For arithmetic, avoid common digits that appear in operands.
        # Pick a target where the digit doesn't contaminate other prompts.
        if family_name == "arithmetic":
            # Pick target with fewest appearances in other prompts' text
            best_target = None
            best_contamination = float("inf")
            for answer, count in answer_counts.most_common():
                if count < 20:
                    continue
                n_contaminated = sum(
                    1 for p in prompts
                    if p.correct_answer != answer and answer in p.prompt_text
                )
                if n_contaminated < best_contamination:
                    best_contamination = n_contaminated
                    best_target = answer
            targets[family_name] = best_target or answer_counts.most_common(1)[0][0]
        else:
            # For other families, pick the most common answer with enough others
            for answer, count in answer_counts.most_common():
                n_other = len(prompts) - count
                if n_other >= 50 and count >= 20:
                    targets[family_name] = answer
                    break
            if family_name not in targets:
                targets[family_name] = answer_counts.most_common(1)[0][0]
    return targets


def get_token_id(tokenizer, word):
    """Get single token ID for a word."""
    for text in [" " + word, word]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else 0


def run_experiment2(args):
    results_dir = Path(args.results_dir) / "exp2"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load Exp1 results
    exp1_dir = Path(args.results_dir) / "exp1"
    with open(exp1_dir / "layer_accuracy.json") as f:
        exp1_lin = json.load(f)
    with open(exp1_dir / "geometric_metrics.json") as f:
        exp1_geo = json.load(f)
    with open(exp1_dir / "mlp_probe_results.json") as f:
        exp1_mlp = json.load(f)

    # Load prompts
    print("=" * 60)
    print("Loading prompts")
    print("=" * 60)
    families = load_families(n_prompts=args.n_prompts)

    # Select steering targets
    targets = select_steering_targets(families)
    for f, t in targets.items():
        n_target = sum(1 for p in families[f] if p.correct_answer == t)
        n_other = len(families[f]) - n_target
        print(f"  {f}: steer toward '{t}' ({n_target} target, {n_other} other)")

    # Load model
    print(f"\nLoading model: {args.model}")
    extractor = ActivationExtractor(
        args.model, device=args.device
    )
    n_layers = extractor.n_layers

    # Build content token mask for accuracy measurement
    import string
    vocab_size = extractor.tokenizer.vocab_size
    punct_and_space = set(string.punctuation + string.whitespace)
    content_mask = torch.ones(vocab_size, dtype=torch.bool)
    for tid in range(vocab_size):
        decoded = extractor.tokenizer.decode([tid])
        if not decoded or all(c in punct_and_space for c in decoded):
            content_mask[tid] = False

    all_results = {}

    for family_name, prompts in families.items():
        print(f"\n{'=' * 60}")
        print(f"Steering: {family_name}")
        print("=" * 60)

        target_answer = targets[family_name]
        target_tid = get_token_id(extractor.tokenizer, target_answer)

        # Split prompts into target (answer matches) and other
        # For arithmetic, filter out other prompts that contain the target digit in text
        target_prompts = [p for p in prompts if p.correct_answer == target_answer]
        if family_name == "arithmetic":
            other_prompts = [p for p in prompts
                           if p.correct_answer != target_answer
                           and target_answer not in p.prompt_text]
        else:
            other_prompts = [p for p in prompts if p.correct_answer != target_answer]

        target_texts = [p.prompt_text for p in target_prompts]
        other_texts = [p.prompt_text for p in other_prompts]

        # We'll steer the "other" prompts toward the target answer
        # and measure how much P(target_token) increases
        print(f"  Target: '{target_answer}' (tid={target_tid})")
        print(f"  Target prompts: {len(target_prompts)}, Other prompts: {len(other_prompts)}")

        # Get baseline probabilities for other prompts
        print("  Computing baseline probs...")
        baseline_probs = extractor.get_next_token_probs(
            other_texts[:args.n_eval], batch_size=args.inference_batch_size
        )
        baseline_target_p = baseline_probs[:, target_tid].cpu().numpy()

        # Extract activations for computing steering directions
        print("  Extracting activations for steering directions...")

        # Load from Exp1 cache if available, otherwise extract
        gpu_cache = exp1_dir / "gpu_cache" / f"{family_name}_gpu.npz"
        if gpu_cache.exists():
            print("    Loading from Exp1 cache...")
            cached = np.load(gpu_cache)
            all_layers = sorted(int(k[4:]) for k in cached.files if k.startswith("act_"))

            # We need to know which prompts in the cache are target vs other
            # The cache has all prompts in the same order as families[family_name]
            target_mask = np.array([p.correct_answer == target_answer for p in prompts])
        else:
            print("    Extracting fresh...")
            all_texts = [p.prompt_text for p in prompts]
            extraction = extractor.extract(
                all_texts, layers=list(range(n_layers)),
                batch_size=args.inference_batch_size
            )
            cached_acts = extraction.to_numpy()
            all_layers = sorted(cached_acts.keys())
            target_mask = np.array([p.correct_answer == target_answer for p in prompts])
            # Wrap in a dict-like for uniform access
            cached = {f"act_{l}": cached_acts[l] for l in all_layers}

        # Steering at each layer
        layer_results = {}
        for layer_idx in all_layers:
            H = cached[f"act_{layer_idx}"]

            # Compute steering direction
            target_acts = H[target_mask]
            other_acts = H[~target_mask]
            direction = target_acts.mean(axis=0) - other_acts.mean(axis=0)
            magnitude = float(np.linalg.norm(direction))

            if magnitude < 1e-10:
                layer_results[str(layer_idx)] = {
                    "effect_size": 0.0, "magnitude": 0.0,
                    "mean_delta_p": 0.0, "collateral": 0.0,
                }
                continue

            direction_unit = direction / magnitude

            # Inject steering at this layer for "other" prompts
            # Use a range of alpha values
            alpha = magnitude * args.steering_alpha

            perturbation = torch.tensor(
                np.tile(direction_unit, (min(len(other_texts), args.n_eval), 1)) * alpha,
                dtype=torch.float32,
            )

            steered_logits = extractor.extract_with_perturbation(
                other_texts[:args.n_eval], layer_idx, perturbation,
                batch_size=args.inference_batch_size,
            )
            steered_probs = torch.softmax(steered_logits.float(), dim=-1)
            steered_target_p = steered_probs[:, target_tid].cpu().numpy()

            # Effect size: increase in P(target) per unit norm
            delta_p = steered_target_p - baseline_target_p[:len(steered_target_p)]
            mean_delta_p = float(np.mean(delta_p))
            effect_size = mean_delta_p / (magnitude + 1e-10)

            # Collateral: how much does the model's accuracy on OTHER correct answers change?
            # Check if the model's top content prediction changed
            baseline_content = baseline_probs[:len(steered_target_p)].clone()
            baseline_content[:, ~content_mask] = 0
            steered_content = steered_probs.clone()
            steered_content[:, ~content_mask] = 0

            correct_tids = [get_token_id(extractor.tokenizer, p.correct_answer)
                           for p in other_prompts[:args.n_eval]]

            baseline_correct = sum(
                baseline_content[i].argmax().item() == correct_tids[i]
                for i in range(len(correct_tids))
            )
            steered_correct = sum(
                steered_content[i].argmax().item() == correct_tids[i]
                for i in range(len(correct_tids))
            )

            collateral = (baseline_correct - steered_correct) / len(correct_tids)

            layer_results[str(layer_idx)] = {
                "effect_size": effect_size,
                "magnitude": magnitude,
                "mean_delta_p": mean_delta_p,
                "collateral": collateral,
                "alpha": alpha,
                "baseline_target_p_mean": float(np.mean(baseline_target_p[:len(steered_target_p)])),
                "steered_target_p_mean": float(np.mean(steered_target_p)),
            }

            print(f"  L{layer_idx:>2}: effect={effect_size:.4f} "
                  f"ΔP={mean_delta_p:+.4f} "
                  f"collateral={collateral:+.3f} "
                  f"‖v‖={magnitude:.1f}", flush=True)

        all_results[family_name] = {
            "target": target_answer,
            "target_tid": target_tid,
            "n_target": int(target_mask.sum()),
            "n_other": int((~target_mask).sum()),
            "layers": layer_results,
        }

    # Free GPU
    del extractor
    gc.collect()
    torch.cuda.empty_cache()

    # ============================================================
    # Correlate with Exp1 LAP metrics
    # ============================================================
    print(f"\n{'=' * 60}")
    print("Correlating steering success with LAP metrics")
    print("=" * 60)

    print("\n=== STEERING RESULTS ===\n")
    header = f"{'Family':<16} {'BestL':>5} {'Effect':>8} {'ΔP':>8} {'Collat':>8}"
    print(header)
    print("-" * len(header))

    for family_name, result in all_results.items():
        layers = result["layers"]
        best_l = max(layers.keys(), key=lambda l: layers[l]["effect_size"])
        best = layers[best_l]
        print(f"{family_name:<16} {best_l:>5} {best['effect_size']:>8.4f} "
              f"{best['mean_delta_p']:>+8.4f} {best['collateral']:>+8.3f}")

    print("\n=== CORRELATION: LAP metrics vs Steering effect size ===\n")
    print(f"{'Family':<16} {'ρ(lin,eff)':>10} {'p':>8} {'ρ(gap,eff)':>10} {'p':>8} "
          f"{'ρ(λ,eff)':>10} {'p':>8}")
    print("-" * 80)

    for family_name in all_results:
        if family_name not in exp1_lin:
            continue

        steer_layers = all_results[family_name]["layers"]
        lin_layers = exp1_lin[family_name]["layers"]
        geo_layers = exp1_geo.get(family_name, {})
        mlp_layers = exp1_mlp.get(family_name, {})

        # Collect per-layer vectors
        common_layers = sorted(set(steer_layers.keys()) & set(lin_layers.keys()), key=int)

        effect_sizes = [steer_layers[l]["effect_size"] for l in common_layers]
        lin_accs = [lin_layers[l]["linear_acc"] for l in common_layers]

        # Probe gap
        gaps = []
        for l in common_layers:
            la = lin_layers[l]["linear_acc"]
            ma = mlp_layers.get(l, {}).get("mlp_acc", 0)
            gaps.append(ma - la)

        # Lambda
        lambdas = [geo_layers.get(l, {}).get("mean_lambda", 0) for l in common_layers]

        rho_lin, p_lin = spearmanr(lin_accs, effect_sizes)
        rho_gap, p_gap = spearmanr(gaps, effect_sizes)
        rho_lam, p_lam = spearmanr(lambdas, effect_sizes)

        print(f"{family_name:<16} {rho_lin:>+10.3f} {p_lin:>8.4f} "
              f"{rho_gap:>+10.3f} {p_gap:>8.4f} "
              f"{rho_lam:>+10.3f} {p_lam:>8.4f}")

    # Print per-layer comparison
    print("\n=== STEERING vs LINEAR ACCURACY BY LAYER ===\n")
    for family_name in all_results:
        if family_name not in exp1_lin:
            continue
        print(f"--- {family_name} (target='{all_results[family_name]['target']}') ---")
        steer_layers = all_results[family_name]["layers"]
        lin_layers = exp1_lin[family_name]["layers"]
        for l in sorted(steer_layers.keys(), key=int):
            eff = steer_layers[l]["effect_size"]
            la = lin_layers.get(l, {}).get("linear_acc", 0)
            bar_l = "#" * int(la * 30)
            bar_e = "+" * int(min(abs(eff) * 300, 30))
            print(f"  L{l:>2}: lin={la:.3f} effect={eff:+.4f} {bar_l} | {bar_e}")
        print()

    # Save results
    with open(results_dir / "steering_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nExperiment 2 (steering) complete. Results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Steering")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Number of 'other' prompts to steer")
    parser.add_argument("--steering-alpha", type=float, default=1.0,
                        help="Steering strength as fraction of direction magnitude")
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_experiment2(args)


if __name__ == "__main__":
    main()
