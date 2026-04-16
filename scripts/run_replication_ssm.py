"""Replication on non-transformer architectures (Mamba, RWKV).

These models have different layer structures, so we adapt the
extraction and unembedding probe accordingly.

Usage:
    python scripts/run_replication_ssm.py --model state-spaces/mamba-1.4b-hf --device cuda
    python scripts/run_replication_ssm.py --model RWKV/v6-Finch-1B6-HF --device cuda
"""

import scripts._env  # noqa: F401

import argparse
import gc
import json
import torch
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from src.data.loader import load_families


def get_token_id(tokenizer, word):
    for text in [" " + word, word]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) == 2:
        return ids[-1]  # SentencePiece fallback
    return ids[0] if ids else 0


@torch.no_grad()
def run_unembedding_probe_generic(model, tokenizer, prompts, correct_tids,
                                   device, batch_size=16):
    """Generic unembedding probe that works with any model supporting
    output_hidden_states=True."""

    # Find the unembedding matrix
    if hasattr(model, 'lm_head'):
        unembed = model.lm_head.weight
    elif hasattr(model, 'head'):
        unembed = model.head.weight
    else:
        raise ValueError("Cannot find unembedding matrix (lm_head or head)")

    # Find the final layer norm
    final_norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm = model.model.norm
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'norm_f'):
        final_norm = model.backbone.norm_f
    elif hasattr(model, 'rwkv') and hasattr(model.rwkv, 'ln_out'):
        final_norm = model.rwkv.ln_out

    # Get number of layers from hidden states
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test forward pass to get layer count
    test_input = tokenizer(["test"], return_tensors="pt", padding=True).to(device)
    test_out = model(**test_input, output_hidden_states=True, return_dict=True)
    n_hidden = len(test_out.hidden_states)
    n_layers = n_hidden - 1  # first is embedding
    print(f"  Detected {n_layers} layers ({n_hidden} hidden states)")

    prompt_texts = [p.prompt_text for p in prompts]
    per_layer_acc = {l: 0 for l in range(n_layers)}
    total = 0

    for batch_start in range(0, len(prompt_texts), batch_size):
        batch = prompt_texts[batch_start:batch_start + batch_size]
        batch_tids = correct_tids[batch_start:batch_start + len(batch)]

        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(device)

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states

        for layer_idx in range(n_layers):
            hs = hidden_states[layer_idx + 1][:, -1, :]  # last position

            if final_norm is not None:
                hs_normed = final_norm(hs)
            else:
                hs_normed = hs

            logits = hs_normed @ unembed.T

            for i in range(len(batch)):
                tid = batch_tids[i]
                if logits[i].argmax().item() == tid:
                    per_layer_acc[layer_idx] += 1

        total += len(batch)

    return {l: acc / total for l, acc in per_layer_acc.items()}, n_layers


def run_replication_ssm(args):
    model_tag = args.model.split("/")[-1]
    results_dir = Path(args.results_dir) / f"replication_{model_tag}"
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_file = results_dir / "results.json"
    if cache_file.exists():
        print("Loading cached results...")
        with open(cache_file) as f:
            results = json.load(f)
        _print_results(results, model_tag)
        return

    # Load prompts
    print("Loading prompts...")
    families = load_families(n_prompts=args.n_prompts)

    # Load model
    print(f"\nLoading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # RWKV and Mamba have norm layers that output fp32, causing dtype mismatches with fp16
    if "rwkv" in args.model.lower() or "mamba" in args.model.lower():
        dtype = torch.float32
    else:
        dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    ).to(args.device)
    model.eval()

    print(f"  Model type: {type(model).__name__}")
    print(f"  Model structure: {[n for n, _ in model.named_children()]}")

    results = {"model": args.model, "families": {}}

    for fam_name, prompts in families.items():
        eval_prompts = prompts[:args.n_eval]
        print(f"\n--- {fam_name} ({len(eval_prompts)} prompts) ---")

        correct_tids = [get_token_id(tokenizer, p.correct_answer)
                       for p in eval_prompts]

        # Model accuracy
        import string
        probs_list = []
        for i in range(0, len(eval_prompts), args.inference_batch_size):
            batch = [p.prompt_text for p in eval_prompts[i:i+args.inference_batch_size]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=128).to(args.device)
            with torch.no_grad():
                out = model(**inputs, return_dict=True)
            logits = out.logits[:, -1, :]
            probs_list.append(torch.softmax(logits.float(), dim=-1).cpu())

        all_probs = torch.cat(probs_list, dim=0)

        vocab_size = all_probs.shape[1]
        punct_and_space = set(string.punctuation + string.whitespace)
        content_mask = torch.ones(vocab_size, dtype=torch.bool)
        for tid in range(vocab_size):
            decoded = tokenizer.decode([tid])
            if not decoded or all(c in punct_and_space for c in decoded):
                content_mask[tid] = False

        content_probs = all_probs.clone()
        content_probs[:, ~content_mask] = 0
        model_acc = float(np.mean([
            content_probs[i].argmax().item() == correct_tids[i]
            for i in range(len(eval_prompts))
        ]))
        print(f"  Model accuracy: {model_acc:.1%}")

        # Unembedding probe
        print("  Unembedding probe...", flush=True)
        try:
            layer_acc, n_layers = run_unembedding_probe_generic(
                model, tokenizer, eval_prompts, correct_tids,
                args.device, batch_size=args.inference_batch_size,
            )
            best_l = max(layer_acc, key=layer_acc.get)
            print(f"  Best: L{best_l} = {layer_acc[best_l]:.3f}")

            results["n_layers"] = n_layers
            results["families"][fam_name] = {
                "model_accuracy": model_acc,
                "layer_accuracy": {str(l): a for l, a in layer_acc.items()},
            }
        except Exception as e:
            print(f"  Failed: {e}")
            results["families"][fam_name] = {
                "model_accuracy": model_acc,
                "error": str(e),
            }

    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    _print_results(results, model_tag)


def _print_results(results, model_tag):
    print(f"\n{'=' * 60}")
    print(f"REPLICATION: {results.get('model', model_tag)}")
    print(f"{'=' * 60}")

    for fam, data in results.get("families", {}).items():
        la = data.get("layer_accuracy", {})
        if not la:
            print(f"\n{fam}: {data.get('error', 'no data')}")
            continue

        best_l = max(la.keys(), key=lambda l: la[l])
        print(f"\n{fam} (model_acc={data['model_accuracy']:.1%}):")

        # Find emergence
        for l in sorted(la.keys(), key=int):
            acc = la[l]
            bar = "#" * int(acc * 30)
            print(f"  L{l:>2}: {acc:.3f} {bar}")

    # Cross-model comparison
    gemma_file = Path("results/exp1/layer_accuracy.json")
    if gemma_file.exists():
        with open(gemma_file) as f:
            gemma = json.load(f)
        print(f"\n--- Comparison with Gemma-2B ---")
        for fam in results.get("families", {}):
            if fam in gemma:
                gl = gemma[fam]["layers"]
                g_best = max(gl.keys(), key=lambda l: gl[l]["linear_acc"])
                g_acc = gl[g_best]["linear_acc"]

                la = results["families"][fam].get("layer_accuracy", {})
                if la:
                    r_best = max(la.keys(), key=lambda l: la[l])
                    r_acc = la[r_best]
                    n_layers = results.get("n_layers", "?")
                    print(f"  {fam:<16} Gemma=L{g_best}({g_acc:.3f}) "
                          f"This=L{r_best}({r_acc:.3f}, {n_layers}L)")


def main():
    parser = argparse.ArgumentParser(description="SSM/RWKV Replication")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=2000)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_replication_ssm(args)


if __name__ == "__main__":
    main()
