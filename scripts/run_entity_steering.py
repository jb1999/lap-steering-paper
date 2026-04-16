"""Demonstrate LAP on entity steering: redirect London→Paris.

Constructs prompts where the correct single-token answer is either
"London" or "Paris". Computes A_lin (logit lens accuracy) at each
layer, then steers London-answer prompts toward answering "Paris".

This validates LAP end-to-end:
  1. A_lin identifies which layers the concept is output-aligned
  2. Steering at high-A_lin layers redirects completions
  3. Steering at low-A_lin layers has no effect
  4. Full generation confirms behavioral change, not just ΔP

Usage:
    python scripts/run_entity_steering_demo.py --device cuda
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


# Prompts where the correct next token is "Paris"
PARIS_PROMPTS = [
    "The capital of France is",
    "The Eiffel Tower is located in",
    "The Louvre museum is in",
    "Notre-Dame cathedral is in",
    "The Seine river flows through",
    "The Arc de Triomphe is in",
    "Croissants are most associated with",
    "The French government is based in",
    "The Champs-Élysées is a famous avenue in",
    "The Moulin Rouge is located in",
    "Montmartre is a historic district in",
    "The Sorbonne university is in",
    "Charles de Gaulle airport serves",
    "The Palace of Versailles is near",
    "The Latin Quarter is a neighborhood in",
    "The Pompidou Centre is in",
    "Sacré-Cœur basilica overlooks",
    "The French Open tennis tournament is held in",
    "Napoleon's tomb is in",
    "The Musée d'Orsay is in",
]

# Prompts where the correct next token is "London"
LONDON_PROMPTS = [
    "The capital of England is",
    "Big Ben is located in",
    "Buckingham Palace is in",
    "The Tower Bridge is in",
    "The Thames river flows through",
    "The British Museum is in",
    "Fish and chips are most associated with",
    "The British government is based in",
    "Oxford Street is a famous shopping street in",
    "The West End theatre district is in",
    "Westminster Abbey is in",
    "Imperial College is in",
    "Heathrow airport serves",
    "The Globe Theatre is in",
    "Soho is a neighborhood in",
    "The Tate Modern is in",
    "St Paul's Cathedral overlooks",
    "Wimbledon tennis tournament is held in",
    "The Crown Jewels are kept in",
    "The National Gallery is in",
]


def run_entity_steering(args):
    results_dir = Path(args.results_dir) / "entity_steering_demo"
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_file = results_dir / "entity_steering_results.json"
    if cache_file.exists():
        print("Loading cached results...")
        with open(cache_file) as f:
            results = json.load(f)
        _print_analysis(results)
        return

    print(f"Loading model: {args.model}")
    extractor = ActivationExtractor(
        args.model, device=args.device,
    )
    n_layers = extractor.n_layers
    print(f"  Layers: {n_layers}, d_model: {extractor.d_model}")

    # Token IDs for target cities
    paris_tid = extractor.tokenizer.encode(" Paris", add_special_tokens=False)
    london_tid = extractor.tokenizer.encode(" London", add_special_tokens=False)
    assert len(paris_tid) == 1 and len(london_tid) == 1, "Expected single tokens"
    paris_tid = paris_tid[0]
    london_tid = london_tid[0]
    print(f"  Token IDs: Paris={paris_tid}, London={london_tid}")

    all_prompts = PARIS_PROMPTS + LONDON_PROMPTS
    labels = np.array([1] * len(PARIS_PROMPTS) + [0] * len(LONDON_PROMPTS))
    # label 1 = Paris, 0 = London

    # ============================================================
    # Step 1: Check baseline model accuracy
    # ============================================================
    print("\nStep 1: Checking baseline accuracy...")
    baseline_probs = extractor.get_next_token_probs(
        all_prompts, batch_size=args.inference_batch_size,
    )

    n_paris_correct = 0
    n_london_correct = 0
    for i, prompt in enumerate(all_prompts):
        top_tid = int(baseline_probs[i].argmax())
        if i < len(PARIS_PROMPTS):
            if top_tid == paris_tid:
                n_paris_correct += 1
        else:
            if top_tid == london_tid:
                n_london_correct += 1

    print(f"  Paris correct: {n_paris_correct}/{len(PARIS_PROMPTS)}")
    print(f"  London correct: {n_london_correct}/{len(LONDON_PROMPTS)}")

    # ============================================================
    # Step 2: Extract activations and compute A_lin at each layer
    # ============================================================
    print("\nStep 2: Extracting activations...")
    extraction = extractor.extract(
        all_prompts, layers=list(range(n_layers)),
        batch_size=args.inference_batch_size,
    )
    activations = extraction.to_numpy()

    print("Step 3: Computing A_lin (logit lens) at each layer...")
    layer_norm = extractor.model.model.norm
    unembed = extractor.model.lm_head

    layer_results = {}
    for l in range(n_layers):
        H = activations[l]
        H_t = torch.tensor(H, dtype=extractor.dtype, device=extractor.device)

        # Apply layer norm + unembedding = logit lens
        with torch.no_grad():
            normed = layer_norm(H_t)
            logits = unembed(normed)  # (n_prompts, vocab)
            preds = logits.argmax(dim=-1).cpu().numpy()

        # A_lin: fraction where logit lens predicts the correct token
        n_correct = 0
        for i in range(len(all_prompts)):
            if i < len(PARIS_PROMPTS):
                if preds[i] == paris_tid:
                    n_correct += 1
            else:
                if preds[i] == london_tid:
                    n_correct += 1
        a_lin = n_correct / len(all_prompts)

        # Also compute direction for steering
        direction = H[labels == 1].mean(axis=0) - H[labels == 0].mean(axis=0)
        magnitude = float(np.linalg.norm(direction))

        layer_results[str(l)] = {
            "a_lin": a_lin,
            "direction_magnitude": magnitude,
        }

        bar = "#" * int(a_lin * 40)
        print(f"  L{l:>2}: A_lin={a_lin:.3f} ‖v‖={magnitude:.1f} {bar}")

    # ============================================================
    # Step 4: Steer London prompts toward Paris
    # ============================================================
    print("\nStep 4: Steering London→Paris at each layer...")

    steering_results = {}
    for l in range(n_layers):
        H = activations[l]
        direction = H[labels == 1].mean(axis=0) - H[labels == 0].mean(axis=0)
        magnitude = float(np.linalg.norm(direction))
        if magnitude < 1e-10:
            continue
        direction_unit = direction / magnitude

        # Add Paris direction to London prompts
        alpha = magnitude * args.steering_alpha
        n_steer = len(LONDON_PROMPTS)
        perturbation = torch.tensor(
            np.tile(direction_unit, (n_steer, 1)) * alpha,
            dtype=torch.float32,
        )

        steered_logits = extractor.extract_with_perturbation(
            LONDON_PROMPTS, l, perturbation,
            batch_size=args.inference_batch_size,
        )
        steered_probs = torch.softmax(steered_logits.float(), dim=-1)

        # Baseline probs for London prompts
        london_baseline = extractor.get_next_token_probs(
            LONDON_PROMPTS, batch_size=args.inference_batch_size,
        )

        p_paris_base = float(london_baseline[:, paris_tid].mean())
        p_paris_steer = float(steered_probs[:, paris_tid].mean())
        p_london_base = float(london_baseline[:, london_tid].mean())
        p_london_steer = float(steered_probs[:, london_tid].mean())

        steering_results[str(l)] = {
            "p_paris_baseline": p_paris_base,
            "p_paris_steered": p_paris_steer,
            "p_london_baseline": p_london_base,
            "p_london_steered": p_london_steer,
            "delta_paris": p_paris_steer - p_paris_base,
            "delta_london": p_london_steer - p_london_base,
            "a_lin": layer_results[str(l)]["a_lin"],
        }

        print(f"  L{l:>2}: ΔP(Paris)={p_paris_steer - p_paris_base:+.4f} "
              f"ΔP(London)={p_london_steer - p_london_base:+.4f} "
              f"A_lin={layer_results[str(l)]['a_lin']:.3f}")

    # Correlation: A_lin vs steering ΔP(Paris)
    if len(steering_results) >= 4:
        a_lins = [steering_results[l]["a_lin"] for l in steering_results]
        deltas = [steering_results[l]["delta_paris"] for l in steering_results]
        rho, p = spearmanr(a_lins, deltas)
        print(f"\n  Correlation(A_lin, ΔP_Paris): ρ={rho:+.3f} (p={p:.6f})")
    else:
        rho, p = 0, 1

    # ============================================================
    # Step 5: Generate completions at middle vs LAP-recommended layer
    # ============================================================
    # Find LAP-recommended layer (highest A_lin)
    lap_layer = max(layer_results.keys(), key=lambda l: layer_results[l]["a_lin"])
    lap_layer_int = int(lap_layer)

    # Middle layer (conventional heuristic from Turner et al., Templeton et al.)
    mid_layer_int = n_layers // 2
    mid_layer = str(mid_layer_int)

    print(f"\nStep 5: Generating steered completions...")
    print(f"  LAP-recommended: L{lap_layer} (A_lin={layer_results[lap_layer]['a_lin']:.3f})")
    print(f"  Middle layer:    L{mid_layer} (A_lin={layer_results[mid_layer]['a_lin']:.3f})")

    demo_prompts = LONDON_PROMPTS[:8]

    # Baseline completions
    baseline_completions = extractor.generate(
        demo_prompts, max_new_tokens=20, batch_size=8,
    )

    generation_examples = []
    for gen_layer_int, gen_label in [(mid_layer_int, "middle"), (lap_layer_int, "LAP")]:
        H_l = activations[gen_layer_int]
        dir_l = H_l[labels == 1].mean(axis=0) - H_l[labels == 0].mean(axis=0)
        mag_l = float(np.linalg.norm(dir_l))
        dir_unit_l = dir_l / mag_l

        for alpha_scale in [0.5, 1.0]:
            steered_gens = extractor.generate_with_perturbation(
                demo_prompts,
                layer=gen_layer_int,
                direction=dir_unit_l,  # add Paris direction
                alpha=mag_l * alpha_scale,
                max_new_tokens=20,
                batch_size=8,
            )

            print(f"\n  {'=' * 70}")
            print(f"  STEERED GENERATION (L{gen_layer_int} [{gen_label}], α={alpha_scale})")
            print(f"  {'=' * 70}")
            for prompt, base_gen, steer_gen in zip(
                demo_prompts, baseline_completions, steered_gens
            ):
                example = {
                    "prompt": prompt,
                    "baseline": base_gen.strip(),
                    "steered": steer_gen.strip(),
                    "alpha": alpha_scale,
                    "layer": gen_layer_int,
                    "layer_type": gen_label,
                }
                generation_examples.append(example)
                print(f"\n  Prompt: {prompt}")
                print(f"  Before: {base_gen.strip()[:80]}")
                print(f"  After:  {steer_gen.strip()[:80]}")

    # Save
    results = {
        "model": args.model,
        "n_paris": len(PARIS_PROMPTS),
        "n_london": len(LONDON_PROMPTS),
        "paris_correct": n_paris_correct,
        "london_correct": n_london_correct,
        "paris_token_id": paris_tid,
        "london_token_id": london_tid,
        "layer_results": layer_results,
        "steering": steering_results,
        "correlation_rho": float(rho),
        "correlation_p": float(p),
        "generation_examples": generation_examples,
        "generation_lap_layer": lap_layer_int,
        "generation_mid_layer": mid_layer_int,
    }

    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)

    del extractor
    gc.collect()
    torch.cuda.empty_cache()

    _print_analysis(results)


def _print_analysis(results):
    print(f"\n{'=' * 60}")
    print("ENTITY STEERING (London→Paris): LAP ANALYSIS")
    print("=" * 60)

    layer_res = results["layer_results"]
    steering = results.get("steering", {})

    best_l = max(layer_res.keys(), key=lambda l: layer_res[l]["a_lin"])
    best_alin = layer_res[best_l]["a_lin"]

    print(f"\nModel: {results['model']}")
    print(f"Paris correct: {results['paris_correct']}/{results['n_paris']}")
    print(f"London correct: {results['london_correct']}/{results['n_london']}")
    print(f"Peak A_lin: L{best_l} = {best_alin:.3f}")

    if steering:
        print(f"\nCorrelation(A_lin, ΔP_Paris): ρ={results.get('correlation_rho', 0):+.3f} "
              f"(p={results.get('correlation_p', 1):.6f})")

        print(f"\n  {'Layer':>5} {'A_lin':>6} {'ΔP(Paris)':>10} {'ΔP(London)':>11}")
        print(f"  {'-' * 35}")
        for l in sorted(steering.keys(), key=int):
            s = steering[l]
            print(f"  L{l:>2}   {s['a_lin']:.3f} {s['delta_paris']:>+10.4f} "
                  f"{s['delta_london']:>+11.4f}")


def main():
    parser = argparse.ArgumentParser(description="Entity Steering Demo")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steering-alpha", type=float, default=1.0)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_entity_steering(args)


if __name__ == "__main__":
    main()
