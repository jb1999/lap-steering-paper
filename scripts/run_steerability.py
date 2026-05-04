"""Cross-concept steerability analysis.

For each of the 18 concept families, compute:
1. Peak logit lens accuracy (A_lin) and probe gap (Δ)
2. Steering ΔP at the best layer
3. Correlate probe gap with steerability across families

This tests the claim: "concepts with small probe gaps are more steerable."

Usage:
    python scripts/run_cross_concept.py --model google/gemma-2-2b --device cuda
"""

import scripts._env  # noqa: F401

import argparse
import gc
import json
import string
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.stats import spearmanr

from src.data.loader import load_families


def get_token_id(tokenizer, word):
    for text in [" " + word, word]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) == 2:
        return ids[-1]
    return ids[0] if ids else 0


def is_single_token(tokenizer, word):
    for text in [" " + word, word]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Cross-concept steerability analysis")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    model_tag = args.model.split("/")[-1]
    save_dir = results_dir / "cross_concept" / model_tag
    save_dir.mkdir(parents=True, exist_ok=True)

    cache_file = save_dir / "results.json"
    if cache_file.exists() and not args.force:
        print("Loading cached results...")
        with open(cache_file) as f:
            results = json.load(f)
        _print_results(results)
        return

    # Load model
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    model.config.use_cache = False

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    # Get final norm + unembed for logit lens
    # Try common model structures
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        final_norm = model.gpt_neox.final_layer_norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        final_norm = model.transformer.ln_f
    else:
        raise ValueError(f"Cannot find final norm in {type(model)}")

    unembed = model.lm_head.weight if hasattr(model, "lm_head") else model.embed_out.weight

    # Get model layers for steering hooks
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model_layers = model.model.layers
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        model_layers = model.gpt_neox.layers
    else:
        raise ValueError(f"Cannot find layers in {type(model)}")

    # Load all families including extended
    print("Loading concept families...")
    families = load_families(n_prompts=args.n_prompts,
                                       include_controlled=True)

    results = {"model": args.model, "n_layers": n_layers, "families": {}}

    for fam_name, prompts in sorted(families.items()):
        # Filter to single-token answers only
        st_prompts = [p for p in prompts if is_single_token(tokenizer, p.correct_answer)]
        if len(st_prompts) < 10:
            print(f"\n  {fam_name}: skipping ({len(st_prompts)} single-token prompts)")
            continue

        eval_prompts = st_prompts[:500]
        texts = [p.prompt_text for p in eval_prompts]
        correct_tids = [get_token_id(tokenizer, p.correct_answer) for p in eval_prompts]

        print(f"\n{'=' * 55}")
        print(f"{fam_name} ({len(eval_prompts)} prompts)")
        print(f"{'=' * 55}")

        # --- Model accuracy ---
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), args.batch_size):
                batch = texts[i:i + args.batch_size]
                inp = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=128).to(args.device)
                # Fix position_ids for left-padded RoPE inputs
                inp["position_ids"] = (inp["attention_mask"].cumsum(-1) - 1).clamp(min=0)
                out = model(**inp, return_dict=True)
                probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1).cpu()
                all_probs.append(probs)

        all_probs = torch.cat(all_probs, dim=0)
        vocab_size = all_probs.shape[1]

        # Content token mask
        punct_space = set(string.punctuation + string.whitespace)
        content_mask = torch.ones(vocab_size, dtype=torch.bool)
        for tid in range(min(vocab_size, 50000)):  # check first 50k tokens
            decoded = tokenizer.decode([tid])
            if not decoded or all(c in punct_space for c in decoded):
                content_mask[tid] = False

        content_probs = all_probs.clone()
        content_probs[:, ~content_mask] = 0
        model_acc = float(np.mean([
            content_probs[i].argmax().item() == correct_tids[i]
            for i in range(len(eval_prompts))
        ]))
        print(f"  Model accuracy: {model_acc:.1%}")

        # --- Extract hidden states + logit lens ---
        all_hidden = {l: [] for l in range(n_layers)}
        with torch.no_grad():
            for i in range(0, len(texts), args.batch_size):
                batch = texts[i:i + args.batch_size]
                inp = tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=128).to(args.device)
                # Fix position_ids for left-padded RoPE inputs
                inp["position_ids"] = (inp["attention_mask"].cumsum(-1) - 1).clamp(min=0)
                out = model(**inp, output_hidden_states=True, return_dict=True)
                for l in range(n_layers):
                    all_hidden[l].append(out.hidden_states[l + 1][:, -1, :].cpu())

        for l in range(n_layers):
            all_hidden[l] = torch.cat(all_hidden[l], dim=0)

        # Logit lens accuracy per layer
        raw_acc = {}
        with torch.no_grad():
            for l in range(n_layers):
                H = all_hidden[l].to(args.device)
                h_normed = final_norm(H)
                logits = h_normed.float() @ unembed.T.float()
                raw_acc[l] = sum(
                    logits[i].argmax().item() == correct_tids[i]
                    for i in range(len(eval_prompts))
                ) / len(eval_prompts)

        best_layer = max(raw_acc, key=raw_acc.get)
        best_alin = raw_acc[best_layer]
        print(f"  Best A_lin: {best_alin:.3f} (L{best_layer})")

        # --- Steering ---
        # Find a steering target with enough prompts
        answer_counts = Counter(p.correct_answer for p in eval_prompts)
        target = None
        for ans, count in answer_counts.most_common():
            if count >= 5:
                target = ans
                break

        steering_dp = {}
        n_target = 0
        if target:
            target_tid = get_token_id(tokenizer, target)
            target_mask = np.array([p.correct_answer == target for p in eval_prompts])
            if fam_name == "arithmetic":
                other_mask = np.array([
                    p.correct_answer != target and target not in p.prompt_text
                    for p in eval_prompts
                ])
            else:
                other_mask = ~target_mask

            n_target = int(target_mask.sum())
            n_other = int(other_mask.sum())

            if n_target >= 3 and n_other >= 10:
                other_texts = [texts[i] for i in range(len(texts)) if other_mask[i]][:200]
                n_steer = len(other_texts)
                print(f"  Steering target='{target}' (n={n_target}), evaluating on {n_steer}")

                for l in range(n_layers):
                    H = all_hidden[l].float().numpy()
                    direction = H[target_mask].mean(axis=0) - H[other_mask].mean(axis=0)
                    magnitude = float(np.linalg.norm(direction))
                    direction_unit = direction / (magnitude + 1e-10)

                    alpha = magnitude * 1.0
                    perturbation = torch.tensor(
                        np.tile(direction_unit, (n_steer, 1)) * alpha,
                        dtype=model.dtype,
                    )

                    # Baseline P(target)
                    baseline_probs = []
                    for i in range(0, n_steer, args.batch_size):
                        batch = other_texts[i:i + args.batch_size]
                        inp = tokenizer(batch, return_tensors="pt", padding=True,
                                        truncation=True, max_length=128).to(args.device)
                        # Fix position_ids for left-padded models (RoPE sensitivity)
                        inp["position_ids"] = (inp["attention_mask"].cumsum(-1) - 1).clamp(min=0)
                        with torch.no_grad():
                            out = model(**inp, return_dict=True)
                        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
                        baseline_probs.append(probs[:, target_tid].cpu())
                    baseline_p = torch.cat(baseline_probs).numpy()

                    # Steered P(target)
                    steered_probs = []
                    for i in range(0, n_steer, args.batch_size):
                        batch = other_texts[i:i + args.batch_size]
                        batch_pert = perturbation[i:i + len(batch)].to(args.device)
                        inp = tokenizer(batch, return_tensors="pt", padding=True,
                                        truncation=True, max_length=128).to(args.device)
                        # Fix position_ids for left-padded models (RoPE sensitivity)
                        inp["position_ids"] = (inp["attention_mask"].cumsum(-1) - 1).clamp(min=0)
                        # With left-padding, last real token is always at the rightmost position
                        seq_len = inp["input_ids"].shape[1]
                        positions = torch.full((len(batch),), seq_len - 1, device=args.device)

                        def make_hook(pert, pos):
                            def hook_fn(module, input, output):
                                hs = output[0] if isinstance(output, tuple) else output
                                for j, p in enumerate(pos):
                                    hs[j, p] += pert[j]
                                return (hs,) + output[1:] if isinstance(output, tuple) else hs
                            return hook_fn

                        handle = model_layers[l].register_forward_hook(
                            make_hook(batch_pert, positions)
                        )
                        with torch.no_grad():
                            out = model(**inp, return_dict=True)
                        handle.remove()

                        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
                        steered_probs.append(probs[:, target_tid].cpu())

                    steered_p = torch.cat(steered_probs).numpy()
                    steering_dp[l] = float(steered_p.mean() - baseline_p.mean())

                best_steer_layer = max(steering_dp, key=steering_dp.get)
                max_dp = steering_dp[best_steer_layer]
                print(f"  Max steering ΔP: {max_dp:+.4f} (L{best_steer_layer})")

                # Steering correlation with A_lin
                layers = sorted(steering_dp.keys())
                rho, p_val = spearmanr(
                    [raw_acc[l] for l in layers],
                    [steering_dp[l] for l in layers],
                )
                print(f"  ρ(A_lin, ΔP): {rho:+.3f} (p={p_val:.4f})")
            else:
                print(f"  Steering: insufficient targets (n_target={n_target}, n_other={n_other})")
        else:
            print(f"  Steering: no answer with >= 5 prompts")

        results["families"][fam_name] = {
            "n_prompts": len(eval_prompts),
            "model_accuracy": model_acc,
            "best_alin": best_alin,
            "best_alin_layer": best_layer,
            "raw_acc": {str(l): float(v) for l, v in raw_acc.items()},
            "steering_target": target,
            "n_target": n_target,
            "steering_dp": {str(l): float(v) for l, v in steering_dp.items()},
            "max_steering_dp": float(max(steering_dp.values())) if steering_dp else None,
            "steering_rho": float(rho) if steering_dp else None,
        }

        del all_hidden
        torch.cuda.empty_cache()

    # Save
    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    _print_results(results)


def _print_results(results):
    print(f"\n{'=' * 80}")
    print(f"CROSS-CONCEPT STEERABILITY: {results['model']}")
    print(f"{'=' * 80}")

    print(f"\n{'Family':<16} {'N':>4} {'Acc':>6} {'A_lin':>6} {'Layer':>6} "
          f"{'Max ΔP':>8} {'ρ(A,ΔP)':>8}")
    print("-" * 60)

    families_with_steering = []
    for fam, r in sorted(results["families"].items()):
        dp = r.get("max_steering_dp")
        rho = r.get("steering_rho")
        print(f"{fam:<16} {r['n_prompts']:>4} {r['model_accuracy']:>5.1%} "
              f"{r['best_alin']:>6.3f} L{r['best_alin_layer']:>3} "
              f"{'---' if dp is None else f'{dp:>+7.4f}'} "
              f"{'---' if rho is None else f'{rho:>+7.3f}'}")
        if dp is not None:
            families_with_steering.append((fam, r))

    if len(families_with_steering) >= 5:
        # Cross-concept correlation: probe gap vs max steerability
        names = [f for f, _ in families_with_steering]
        alins = [r["best_alin"] for _, r in families_with_steering]
        dps = [r["max_steering_dp"] for _, r in families_with_steering]
        gaps = [r["model_accuracy"] - r["best_alin"] for _, r in families_with_steering]

        rho_alin_dp, p_alin = spearmanr(alins, dps)
        print(f"\nCross-concept (n={len(families_with_steering)}):")
        print(f"  ρ(A_lin, max_ΔP) = {rho_alin_dp:+.3f} (p={p_alin:.4f})")
        print(f"  Higher A_lin → {'more' if rho_alin_dp > 0 else 'less'} steerable")


if __name__ == "__main__":
    main()
