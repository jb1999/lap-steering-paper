"""Experiment 1: LAP Across Layers (H1)

For each concept family, measure linear accessibility at each layer
by applying the model's own unembedding matrix to intermediate hidden states.

If the correct answer token has the highest logit at layer L, the concept
is "linearly accessible" at that layer — the model's linear readout is
sufficient without further nonlinear processing.

Pipeline:
  Phase 0: Load prompts, determine correct token IDs
  Phase 1 (GPU): Extract hidden states + perturbation sensitivity
  Phase 1b (GPU): Apply unembedding at each layer, compute per-layer accuracy
  Phase 2 (CPU): Compute LAP geometric metrics
  Phase 3: Report results, generate figures

Usage:
    python scripts/run_experiment1.py --model google/gemma-2-2b --device cuda
"""

import scripts._env  # noqa: F401 — must be first

import argparse
import gc
import json
import pickle
import torch
import numpy as np
from pathlib import Path

from src.extraction.activations import ActivationExtractor
from src.data.loader import load_families
from src.metrics.lap import LAPMetrics
from src.analysis.figures import FigureGenerator


@torch.no_grad()
def compute_linear_accuracy_all_layers(
    extractor,
    prompts,
    correct_token_ids,
    batch_size=32,
):
    """Apply model's unembedding matrix to hidden states at each layer.

    Returns:
        per_layer_acc: dict mapping layer -> accuracy (fraction of prompts
            where the correct token has the highest logit)
        per_layer_ranks: dict mapping layer -> array of ranks of the correct
            token in the logit distribution (1 = highest)
    """
    # Get the unembedding matrix (lm_head)
    unembed = extractor.model.lm_head.weight  # (vocab_size, d_model)

    # Also get the final layer norm
    final_norm = extractor.model.model.norm

    all_layers = list(range(extractor.n_layers))
    n_prompts = len(prompts)

    # Collect per-layer logits for the correct token
    correct_logits = {l: [] for l in all_layers}
    correct_ranks = {l: [] for l in all_layers}

    prompt_texts = [p.prompt_text for p in prompts]

    for batch_start in range(0, n_prompts, batch_size):
        batch_prompts = prompt_texts[batch_start:batch_start + batch_size]
        batch_correct = correct_token_ids[batch_start:batch_start + len(batch_prompts)]
        bs = len(batch_prompts)

        inputs = extractor.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(extractor.device)

        outputs = extractor.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states  # (n_layers+1) x (batch, seq, d_model)

        for layer_idx in all_layers:
            # Get hidden state at last position
            hs = hidden_states[layer_idx + 1]  # (batch, seq, d_model)
            # With left-padding, last token is at position -1
            hs_last = hs[:, -1, :]  # (batch, d_model)

            # Apply layer norm then unembedding
            hs_normed = final_norm(hs_last)  # (batch, d_model)
            logits = hs_normed @ unembed.T  # (batch, vocab_size)

            for i in range(bs):
                tid = batch_correct[i]
                # Rank of correct token (1 = best)
                rank = (logits[i] >= logits[i, tid]).sum().item()
                correct_ranks[layer_idx].append(rank)
                correct_logits[layer_idx].append(logits[i, tid].item())

        if batch_start % (batch_size * 5) == 0:
            print(f"    Batch {batch_start // batch_size + 1}/"
                  f"{(n_prompts + batch_size - 1) // batch_size}...", end="\r")

    print()

    # Compute accuracy (rank == 1 means correct token is top prediction)
    per_layer_acc = {}
    per_layer_rank_arrays = {}
    for l in all_layers:
        ranks = np.array(correct_ranks[l])
        per_layer_acc[l] = float(np.mean(ranks == 1))
        per_layer_rank_arrays[l] = ranks

    return per_layer_acc, per_layer_rank_arrays


def get_correct_token_ids(extractor_or_tokenizer, prompts,
                          tokenizer_name=None, cache_dir=None):
    """Get the token ID of each prompt's correct answer.

    Args:
        extractor_or_tokenizer: ActivationExtractor, tokenizer, or None.
        prompts: List of ProbePrompts.
        tokenizer_name: Model name to load tokenizer from (if extractor is None).
        cache_dir: HF cache dir.

    Returns list of token IDs (one per prompt).
    """
    if extractor_or_tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    elif hasattr(extractor_or_tokenizer, 'tokenizer'):
        tokenizer = extractor_or_tokenizer.tokenizer
    else:
        tokenizer = extractor_or_tokenizer

    token_ids = []
    for p in prompts:
        answer = p.correct_answer
        ids_space = tokenizer.encode(" " + answer, add_special_tokens=False)
        ids_bare = tokenizer.encode(answer, add_special_tokens=False)

        if len(ids_space) == 1:
            token_ids.append(ids_space[0])
        elif len(ids_bare) == 1:
            token_ids.append(ids_bare[0])
        else:
            token_ids.append(ids_bare[0] if ids_bare else 0)

    return token_ids


def run_experiment1(args):
    results_dir = Path(args.results_dir) / "exp1"
    results_dir.mkdir(parents=True, exist_ok=True)
    gpu_cache_dir = results_dir / "gpu_cache"
    gpu_cache_dir.mkdir(exist_ok=True)

    # ============================================================
    # PHASE 0: Load prompts
    # ============================================================
    print("=" * 60)
    print("PHASE 0: Loading probe prompts")
    print("=" * 60)

    families = load_families(n_prompts=args.n_prompts)

    # ============================================================
    # PHASE 1: GPU — everything that needs the model
    # ============================================================
    print(f"\n{'=' * 60}")
    print("PHASE 1: GPU — extract, evaluate, compute linear accuracy")
    print("=" * 60)

    # Check what's cached
    results_cache = results_dir / "layer_accuracy.json"
    if results_cache.exists():
        print("  Loading cached layer accuracy results...")
        with open(results_cache) as f:
            all_results = json.load(f)
        # Check if all families are present
        missing = [f for f in families if f not in all_results]
        if not missing:
            print("  All families cached.")
        else:
            print(f"  Missing: {missing}")
    else:
        all_results = {}
        missing = list(families.keys())

    if missing:
        print(f"\nLoading model: {args.model}")
        extractor = ActivationExtractor(
            args.model, device=args.device
        )
        print(f"  Layers: {extractor.n_layers}, d_model: {extractor.d_model}")

        for family_name in missing:
            prompts = families[family_name]
            print(f"\n--- {family_name} ({len(prompts)} prompts) ---")

            # Get correct token IDs
            correct_token_ids = get_correct_token_ids(extractor, prompts)

            # Report token ID stats
            unique_tids = set(correct_token_ids)
            print(f"  Unique answer tokens: {len(unique_tids)}")

            # Model accuracy: is the correct token the top content token?
            print("  Evaluating model accuracy...")
            prompt_texts = [p.prompt_text for p in prompts]
            model_probs = extractor.get_next_token_probs(prompt_texts, batch_size=args.inference_batch_size)
            correct_arr = np.array(correct_token_ids)

            # Build mask: exclude whitespace-only and punctuation-only tokens
            if not hasattr(run_experiment1, '_content_mask'):
                import string
                vocab_size = model_probs.shape[1]
                content_mask = torch.ones(vocab_size, dtype=torch.bool)
                punct_and_space = set(string.punctuation + string.whitespace)
                # Cache content_mask on disk per tokenizer — vocab is fixed.
                tok_name = getattr(extractor.tokenizer, "name_or_path", "tokenizer").replace("/", "_")
                mask_cache = gpu_cache_dir / f"content_mask_{tok_name}_v{vocab_size}.pt"
                if mask_cache.exists():
                    content_mask = torch.load(mask_cache)
                else:
                    # Single batched decode via the fast Rust tokenizer — orders of
                    # magnitude faster than per-id decode() in a Python loop.
                    all_decoded = extractor.tokenizer.batch_decode(
                        [[tid] for tid in range(vocab_size)]
                    )
                    for tid, decoded in enumerate(all_decoded):
                        if not decoded or all(c in punct_and_space for c in decoded):
                            content_mask[tid] = False
                    torch.save(content_mask, mask_cache)
                run_experiment1._content_mask = content_mask
                n_excluded = vocab_size - content_mask.sum().item()
                print(f"  Excluded {n_excluded} space/punct tokens from {vocab_size}")
            content_mask = run_experiment1._content_mask

            # Zero out non-content token probabilities
            content_probs = model_probs.clone()
            content_probs[:, ~content_mask] = 0

            # Rank among content tokens only
            model_ranks = []
            for i in range(len(prompts)):
                rank = int((content_probs[i] >= content_probs[i, correct_arr[i]]).sum().item())
                model_ranks.append(rank)
            model_ranks = np.array(model_ranks)

            top1_accuracy = float(np.mean(model_ranks == 1))
            top5_accuracy = float(np.mean(model_ranks <= 5))
            median_rank = float(np.median(model_ranks))
            model_correct_mask = (model_ranks == 1).tolist()

            print(f"  Model accuracy (content only): top1={top1_accuracy:.1%}, "
                  f"top5={top5_accuracy:.1%}, median_rank={median_rank:.0f}")

            # Linear accuracy at each layer (unembedding probe)
            print("  Computing linear accuracy at each layer...")
            per_layer_acc, per_layer_ranks = compute_linear_accuracy_all_layers(
                extractor, prompts, correct_token_ids,
                batch_size=args.inference_batch_size,
            )

            # Breakdown: accuracy on model-correct vs model-incorrect
            breakdown = {}
            for l in sorted(per_layer_acc.keys()):
                ranks = per_layer_ranks[l]
                cm = np.array(model_correct_mask)

                acc_all = float(np.mean(ranks == 1))
                acc_correct = float(np.mean(ranks[cm] == 1)) if cm.sum() > 0 else float("nan")
                acc_incorrect = float(np.mean(ranks[~cm] == 1)) if (~cm).sum() > 0 else float("nan")
                median_rank = float(np.median(ranks))

                breakdown[str(l)] = {
                    "linear_acc": acc_all,
                    "linear_acc_model_correct": acc_correct,
                    "linear_acc_model_incorrect": acc_incorrect,
                    "median_rank": median_rank,
                }

            all_results[family_name] = {
                "model_top1": top1_accuracy,
                "model_top5": top5_accuracy,
                "model_median_rank": median_rank,
                "n_prompts": len(prompts),
                "n_unique_tokens": len(unique_tids),
                "layers": breakdown,
            }

        # Save results
        with open(results_cache, "w") as f:
            json.dump(all_results, f, indent=2)

        # Compute perturbation sensitivity if not cached
        for family_name in missing:
            prompts = families[family_name]
            ps_cache = gpu_cache_dir / f"{family_name}_gpu.npz"

            if ps_cache.exists():
                print(f"\n  {family_name}: perturbation sensitivity cached")
                continue

            print(f"\n  {family_name}: computing perturbation sensitivity...")
            prompt_texts = [p.prompt_text for p in prompts]
            all_layers = list(range(extractor.n_layers))

            # Extract activations
            extraction = extractor.extract(
                prompt_texts, layers=all_layers, batch_size=args.inference_batch_size
            )
            activations = extraction.to_numpy()

            perturbation_sensitivities = {}
            for layer_idx in all_layers:
                print(f"    Layer {layer_idx}/{extractor.n_layers - 1}...", end="\r")
                perturbation_sensitivities[layer_idx] = LAPMetrics.perturbation_sensitivity(
                    extractor, prompt_texts, layer_idx,
                    n_perturbations=args.n_perturbations,
                    batch_size=args.inference_batch_size,
                )
            print(f"    Done — {len(all_layers)} layers.              ")

            save_dict = {}
            for l, a in activations.items():
                save_dict[f"act_{l}"] = a
            for l, ps in perturbation_sensitivities.items():
                save_dict[f"ps_{l}"] = ps
            np.savez(ps_cache, **save_dict)

        # Free GPU
        print("\nUnloading model from GPU...")
        del extractor
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # PHASE 2b: Geometric metrics (CPU, from cached activations)
    # ============================================================
    print(f"\n{'=' * 60}")
    print("PHASE 2b: Geometric metrics (from cached activations)")
    print("=" * 60)

    geo_cache = results_dir / "geometric_metrics.json"
    if geo_cache.exists():
        print("  Loading cached geometric metrics...")
        with open(geo_cache) as f:
            geo_results = json.load(f)
    else:
        geo_results = {}

    for family_name in families:
        if family_name in geo_results:
            print(f"  {family_name}: cached")
            continue

        npz_file = gpu_cache_dir / f"{family_name}_gpu.npz"
        if not npz_file.exists():
            print(f"  {family_name}: no activation cache, skipping")
            continue

        print(f"  {family_name}...", end=" ", flush=True)
        cached = np.load(npz_file)

        layers_data = {}
        layer_keys = sorted(int(k[4:]) for k in cached.files if k.startswith("act_"))

        for layer_idx in layer_keys:
            H = cached[f"act_{layer_idx}"]

            # Effective rank (RANKME)
            H_centered = H - H.mean(axis=0)
            sv = np.linalg.svd(H_centered, compute_uv=False)
            sv_pos = sv[sv > 1e-10]
            sv_norm = sv_pos / sv_pos.sum()
            eff_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-15))))

            # PCA concentration
            sv_sq = sv ** 2
            total_var = sv_sq.sum()
            kappa_32 = float(sv_sq[:32].sum() / (total_var + 1e-10))
            kappa_128 = float(sv_sq[:min(128, len(sv_sq))].sum() / (total_var + 1e-10))

            # Condition number (full activation matrix)
            if sv[-1] > 1e-10:
                cond = float(sv[0] / sv[-1])
            else:
                cond = float("inf")

            # Perturbation sensitivity (mean λ)
            ps_key = f"ps_{layer_idx}"
            if ps_key in cached.files:
                lambda_vals = cached[ps_key]
                mean_lambda = float(np.mean(lambda_vals))
                std_lambda = float(np.std(lambda_vals))
            else:
                mean_lambda = float("nan")
                std_lambda = float("nan")

            layers_data[str(layer_idx)] = {
                "effective_rank": eff_rank,
                "pca_kappa_32": kappa_32,
                "pca_kappa_128": kappa_128,
                "condition_number": cond,
                "mean_lambda": mean_lambda,
                "std_lambda": std_lambda,
            }

        geo_results[family_name] = layers_data
        print("done", flush=True)

    with open(geo_cache, "w") as f:
        json.dump(geo_results, f, indent=2)

    # Print geometric metrics summary
    print("\n=== GEOMETRIC METRICS (at best linear accuracy layer) ===\n")
    print(f"{'Family':<18} {'Layer':>5} {'EffRank':>8} {'κ32':>6} {'κ128':>6} "
          f"{'Cond':>10} {'λ_mean':>8} {'λ_std':>8}")
    print("-" * 80)

    for family_name, result in all_results.items():
        layers = result["layers"]
        best_layer = max(layers.keys(), key=lambda l: layers[l]["linear_acc"])
        if family_name in geo_results and best_layer in geo_results[family_name]:
            g = geo_results[family_name][best_layer]
            cond_str = f"{g['condition_number']:.1f}" if g['condition_number'] < 1e6 else "inf"
            print(f"{family_name:<18} {best_layer:>5} {g['effective_rank']:>8.1f} "
                  f"{g['pca_kappa_32']:>6.3f} {g['pca_kappa_128']:>6.3f} "
                  f"{cond_str:>10} {g['mean_lambda']:>8.2f} {g['std_lambda']:>8.2f}")

    # Print λ trajectory
    print("\n=== PERTURBATION SENSITIVITY (λ) BY LAYER ===\n")
    for family_name in families:
        if family_name not in geo_results:
            continue
        gd = geo_results[family_name]
        print(f"--- {family_name} ---")
        for l in sorted(gd.keys(), key=int):
            lam = gd[l]["mean_lambda"]
            if not np.isnan(lam):
                bar = "#" * int(min(lam, 50))
                print(f"  L{l:>2}: {lam:>7.2f} {bar}")
        print()

    # ============================================================
    # PHASE 2c: MLP probe gap (GPU, from cached activations)
    # ============================================================
    print(f"\n{'=' * 60}")
    print("PHASE 2c: MLP probe gap")
    print("=" * 60)

    mlp_cache = results_dir / "mlp_probe_results.json"
    if mlp_cache.exists():
        print("  Loading cached MLP results...")
        with open(mlp_cache) as f:
            mlp_results = json.load(f)
    else:
        mlp_results = {}

    families_needing_mlp = [f for f in families if f not in mlp_results]

    if families_needing_mlp:
        from src.probes.nonlinear_probe import NonlinearProbe
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load model weights for unembedding matrix (needed by NonlinearProbe)
        print("\n  Loading model for unembedding matrix...")
        _model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16
        )
        unembed_weight = _model.lm_head.weight.detach()
        ln_weight = _model.model.norm.weight.detach()
        ln_bias = getattr(_model.model.norm, 'bias', None)
        if ln_bias is not None:
            ln_bias = ln_bias.detach()
        del _model
        gc.collect()

        _tokenizer = AutoTokenizer.from_pretrained(args.model)

        for family_name in families_needing_mlp:
            npz_file = gpu_cache_dir / f"{family_name}_gpu.npz"
            if not npz_file.exists():
                continue

            prompts = families[family_name]
            print(f"\n  {family_name}...")

            cached = np.load(npz_file)
            layer_keys = sorted(int(k[4:]) for k in cached.files if k.startswith("act_"))

            # Get correct token IDs (full vocab, not class-mapped)
            correct_token_ids = get_correct_token_ids(_tokenizer, prompts)
            tids_arr = np.array(correct_token_ids)

            layers_mlp = {}
            for layer_idx in layer_keys:
                H = cached[f"act_{layer_idx}"]

                # Train/test split
                split = int(0.8 * len(H))
                probe = NonlinearProbe(
                    unembed_weight=unembed_weight,
                    layer_norm_weight=ln_weight,
                    layer_norm_bias=ln_bias,
                    device=args.device,
                    max_epochs=50,
                    patience=5,
                )
                probe.fit(H[:split], tids_arr[:split], H[split:], tids_arr[split:])
                mlp_acc = probe.score(H[split:], tids_arr[split:])

                layers_mlp[str(layer_idx)] = {"mlp_acc": mlp_acc}
                print(f"    L{layer_idx:>2}: mlp={mlp_acc:.3f}", flush=True)

            mlp_results[family_name] = layers_mlp

        with open(mlp_cache, "w") as f:
            json.dump(mlp_results, f, indent=2)

        torch.cuda.empty_cache()

    # Print probe gap summary
    print("\n=== PROBE GAP (MLP - Linear) BY LAYER ===\n")
    for family_name in families:
        if family_name not in mlp_results or family_name not in all_results:
            continue
        print(f"--- {family_name} ---")
        layers_lin = all_results[family_name]["layers"]
        layers_mlp = mlp_results[family_name]
        for l in sorted(layers_lin.keys(), key=int):
            lin_acc = layers_lin[l]["linear_acc"]
            mlp_acc = layers_mlp.get(l, {}).get("mlp_acc", 0)
            gap = mlp_acc - lin_acc
            bar_lin = "#" * int(lin_acc * 40)
            bar_mlp = "+" * int(max(0, gap) * 40)
            print(f"  L{l:>2}: lin={lin_acc:.3f} mlp={mlp_acc:.3f} gap={gap:+.3f} {bar_lin}{bar_mlp}")
        print()

    # ============================================================
    # PHASE 3: Final summary & figures
    # ============================================================
    print(f"\n{'=' * 60}")
    print("PHASE 3: Final summary & figures")
    print("=" * 60)

    # Print summary table
    print("\n=== RESULTS SUMMARY ===\n")
    header = (f"{'Family':<18} {'Top1':>6} {'Top5':>6} {'MdRnk':>6} {'#Tok':>5} "
              f"{'BestL':>5} {'LinAcc':>7} {'Acc(a)':>7} {'Acc(b)':>7} {'MdRnk':>6}")
    print(header)
    print("-" * len(header))

    for family_name, result in all_results.items():
        layers = result["layers"]
        best_layer = max(layers.keys(), key=lambda l: layers[l]["linear_acc"])
        best = layers[best_layer]

        print(f"{family_name:<18} "
              f"{result.get('model_top1', 0):>5.0%} "
              f"{result.get('model_top5', 0):>5.0%} "
              f"{result.get('model_median_rank', 0):>6.0f} "
              f"{result['n_unique_tokens']:>5} "
              f"{best_layer:>5} "
              f"{best['linear_acc']:>7.3f} "
              f"{best['linear_acc_model_correct']:>7.3f} "
              f"{best['linear_acc_model_incorrect']:>7.3f} "
              f"{best['median_rank']:>6.0f}")

    # Print per-layer detail for each family
    print("\n=== LINEAR ACCURACY BY LAYER ===\n")
    for family_name, result in all_results.items():
        layers = result["layers"]
        print(f"--- {family_name} ---")
        for l in sorted(layers.keys(), key=int):
            d = layers[l]
            bar = "#" * int(d["linear_acc"] * 50)
            print(f"  L{l:>2}: {d['linear_acc']:.3f} {bar}")
        print()

    # Save summary
    with open(results_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nExperiment 1 complete. Results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: LAP Across Layers")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=2000)
    parser.add_argument("--n-perturbations", type=int, default=10)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_experiment1(args)


if __name__ == "__main__":
    main()
