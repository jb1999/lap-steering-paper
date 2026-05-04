"""Replication: Run core LAP measurements on a different model.

Runs Exp1 (unembedding probe + MLP probe) and Exp2 (steering)
on a specified model to test whether LAP patterns generalize.

Usage:
    python scripts/run_replication.py --model meta-llama/Llama-3.1-8B --device cuda
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


def get_token_id(tokenizer, word):
    for text in [" " + word, word]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    # Fallback for SentencePiece tokenizers (e.g., Mistral) that prepend ▁ to
    # digits, making them 2 tokens. Use the last token (the actual character).
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) == 2:
        return ids[-1]
    return ids[0] if ids else 0


@torch.no_grad()
def run_unembedding_probe(extractor, prompts, correct_tids, batch_size=32):
    """Unembedding probe at each layer. Returns per-layer accuracy."""
    unembed = extractor.model.lm_head.weight
    final_norm = extractor.model.model.norm
    n_layers = extractor.n_layers

    prompt_texts = [p.prompt_text for p in prompts]
    per_layer_correct = {l: 0 for l in range(n_layers)}
    total = 0

    for batch_start in range(0, len(prompt_texts), batch_size):
        batch = prompt_texts[batch_start:batch_start + batch_size]
        batch_tids = correct_tids[batch_start:batch_start + len(batch)]

        inputs = extractor.tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(extractor.device)
        # Fix position_ids for left-padded RoPE inputs
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp(min=0)

        outputs = extractor.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states

        for layer_idx in range(n_layers):
            hs = hidden_states[layer_idx + 1][:, -1, :]
            hs_normed = final_norm(hs)
            logits = hs_normed @ unembed.T

            for i in range(len(batch)):
                if logits[i].argmax().item() == batch_tids[i]:
                    per_layer_correct[layer_idx] += 1

        total += len(batch)

    return {l: c / total for l, c in per_layer_correct.items()}


@torch.no_grad()
def run_steering(extractor, prompts, target_answer, batch_size=32, n_eval=200):
    """Steering at each layer. Returns per-layer ΔP."""
    n_layers = extractor.n_layers
    target_tid = get_token_id(extractor.tokenizer, target_answer)

    target_prompts = [p for p in prompts if p.correct_answer == target_answer]
    other_prompts = [p for p in prompts if p.correct_answer != target_answer
                     and target_answer not in p.prompt_text]

    if len(target_prompts) < 5 or len(other_prompts) < 20:
        return None, "too_few_prompts"

    other_texts = [p.prompt_text for p in other_prompts[:n_eval]]

    # Baseline P(target) on other prompts
    baseline_probs = extractor.get_next_token_probs(other_texts, batch_size=batch_size)
    baseline_target_p = baseline_probs[:, target_tid].cpu().numpy()

    # Extract activations for direction computation
    all_texts = [p.prompt_text for p in prompts]
    target_mask = np.array([p.correct_answer == target_answer for p in prompts])

    sample_layers = list(range(n_layers))

    extraction = extractor.extract(all_texts, layers=sample_layers, batch_size=batch_size)
    activations = extraction.to_numpy()

    layer_results = {}
    for layer_idx in sample_layers:
        H = activations[layer_idx]
        direction = H[target_mask].mean(axis=0) - H[~target_mask].mean(axis=0)
        magnitude = float(np.linalg.norm(direction))
        if magnitude < 1e-10:
            layer_results[layer_idx] = 0.0
            continue

        direction_unit = direction / magnitude
        perturbation = torch.tensor(
            np.tile(direction_unit, (len(other_texts), 1)) * magnitude,
            dtype=torch.float32,
        )

        steered_logits = extractor.extract_with_perturbation(
            other_texts, layer_idx, perturbation, batch_size=batch_size,
        )
        steered_probs = torch.softmax(steered_logits.float(), dim=-1)
        steered_target_p = steered_probs[:, target_tid].cpu().numpy()

        dp = float(np.mean(steered_target_p - baseline_target_p[:len(steered_target_p)]))
        layer_results[layer_idx] = dp

    return layer_results, "ok"


def run_replication(args):
    model_tag = args.model.split("/")[-1]
    results_dir = Path(args.results_dir) / f"replication_{model_tag}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    results_cache = results_dir / "results.json"
    if results_cache.exists():
        print("Loading cached results...")
        with open(results_cache) as f:
            all_results = json.load(f)
        _print_results(all_results, model_tag)
        return

    # Load prompts
    print("Loading prompts...")
    families = load_families(n_prompts=args.n_prompts)

    # Load model
    print(f"\nLoading model: {args.model}")
    extractor = ActivationExtractor(
        args.model, device=args.device
    )
    n_layers = extractor.n_layers
    print(f"  Layers: {n_layers}, d_model: {extractor.d_model}")

    # Select steering targets (same logic as exp2)
    from collections import Counter
    targets = {}
    for family_name, prompts in families.items():
        counts = Counter(p.correct_answer for p in prompts)
        # Pick answer with least contamination
        best, best_contam = None, float("inf")
        for ans, c in counts.most_common():
            if c < 10:
                continue
            contam = sum(1 for p in prompts if p.correct_answer != ans and ans in p.prompt_text)
            if contam < best_contam:
                best_contam = contam
                best = ans
        targets[family_name] = best or counts.most_common(1)[0][0]

    all_results = {"model": args.model, "n_layers": n_layers, "d_model": extractor.d_model,
                   "families": {}}

    for family_name, prompts in families.items():
        print(f"\n{'=' * 50}")
        print(f"{family_name} ({len(prompts)} prompts)")
        print("=" * 50)

        eval_prompts = prompts[:args.n_eval]
        correct_tids = [get_token_id(extractor.tokenizer, p.correct_answer) for p in eval_prompts]

        # Model accuracy
        probs = extractor.get_next_token_probs(
            [p.prompt_text for p in eval_prompts], batch_size=args.inference_batch_size
        )
        import string
        vocab_size = probs.shape[1]
        punct_and_space = set(string.punctuation + string.whitespace)
        content_mask = torch.ones(vocab_size, dtype=torch.bool)
        for tid in range(vocab_size):
            decoded = extractor.tokenizer.decode([tid])
            if not decoded or all(c in punct_and_space for c in decoded):
                content_mask[tid] = False

        content_probs = probs.clone()
        content_probs[:, ~content_mask] = 0
        model_acc = float(np.mean([
            content_probs[i].argmax().item() == correct_tids[i]
            for i in range(len(eval_prompts))
        ]))
        print(f"  Model accuracy: {model_acc:.1%}")

        # Unembedding probe
        print("  Unembedding probe...", flush=True)
        layer_acc = run_unembedding_probe(extractor, eval_prompts, correct_tids,
                                          batch_size=args.inference_batch_size)
        best_l = max(layer_acc, key=layer_acc.get)
        print(f"  Best: L{best_l} = {layer_acc[best_l]:.3f}")

        # Steering
        target = targets[family_name]
        print(f"  Steering (target='{target}')...", flush=True)
        steer_results, status = run_steering(
            extractor, prompts, target,
            batch_size=args.inference_batch_size, n_eval=args.n_eval,
        )

        if steer_results:
            best_steer_l = max(steer_results, key=steer_results.get)
            print(f"  Best steering: L{best_steer_l} = {steer_results[best_steer_l]:+.4f}")

        all_results["families"][family_name] = {
            "model_accuracy": model_acc,
            "target": target,
            "layer_accuracy": {str(l): a for l, a in layer_acc.items()},
            "steering": {str(l): dp for l, dp in steer_results.items()} if steer_results else {},
        }

    # Save
    with open(results_cache, "w") as f:
        json.dump(all_results, f, indent=2)

    del extractor
    gc.collect()
    torch.cuda.empty_cache()

    _print_results(all_results, model_tag)


def _print_results(all_results, model_tag):
    """Print and analyze replication results."""
    n_layers = all_results["n_layers"]

    print(f"\n{'=' * 70}")
    print(f"REPLICATION RESULTS: {all_results['model']}")
    print(f"Layers: {n_layers}, d_model: {all_results['d_model']}")
    print("=" * 70)

    # Linear accuracy trajectory
    print("\n=== LINEAR ACCURACY BY LAYER ===\n")
    for family, result in all_results["families"].items():
        la = result["layer_accuracy"]
        print(f"--- {family} (model_acc={result['model_accuracy']:.1%}) ---")
        for l in sorted(la.keys(), key=int):
            acc = la[l]
            bar = "#" * int(acc * 40)
            print(f"  L{l:>2}: {acc:.3f} {bar}")
        print()

    # Steering trajectory
    print("=== STEERING ΔP BY LAYER ===\n")
    for family, result in all_results["families"].items():
        steer = result.get("steering", {})
        if not steer:
            print(f"--- {family}: no steering data ---\n")
            continue
        target = result.get("target", "?")
        print(f"--- {family} (target='{target}') ---")
        for l in sorted(steer.keys(), key=int):
            dp = steer[l]
            bar = "+" * int(max(0, dp) * 40)
            print(f"  L{l:>2}: {dp:+.4f} {bar}")
        print()

    # Correlation
    print("=== CORRELATION: Linear Acc vs Steering ===\n")
    for family, result in all_results["families"].items():
        la = result["layer_accuracy"]
        steer = result.get("steering", {})
        common = sorted(set(la.keys()) & set(steer.keys()), key=int)
        if len(common) < 5:
            continue
        lin_vals = [la[l] for l in common]
        steer_vals = [steer[l] for l in common]
        rho, p = spearmanr(lin_vals, steer_vals)
        print(f"  {family:<16} ρ(lin, ΔP) = {rho:+.3f} (p={p:.4f})")

    # Compare with Gemma-2B
    gemma_file = Path("results/exp1/layer_accuracy.json")
    if gemma_file.exists():
        with open(gemma_file) as f:
            gemma = json.load(f)
        print("\n=== CROSS-MODEL COMPARISON ===\n")
        print(f"{'Family':<16} {'Gemma-2B best':>15} {'This model best':>15}")
        print("-" * 50)
        for family in all_results["families"]:
            if family in gemma:
                g_layers = gemma[family]["layers"]
                g_best = max(g_layers.keys(), key=lambda l: g_layers[l]["linear_acc"])
                g_acc = g_layers[g_best]["linear_acc"]

                r_layers = all_results["families"][family]["layer_accuracy"]
                r_best = max(r_layers.keys(), key=lambda l: r_layers[l])
                r_acc = r_layers[r_best]

                print(f"{family:<16} L{g_best}={g_acc:.3f} (26L) L{r_best}={r_acc:.3f} ({n_layers}L)")

    print(f"\nReplication complete.")


def main():
    parser = argparse.ArgumentParser(description="LAP Replication on new model")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_replication(args)


if __name__ == "__main__":
    main()
