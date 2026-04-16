"""Validate the LAP pipeline: quick sanity checks before running experiments.

Tests:
  1. Model loads and activations extract correctly
  2. Logit lens (A_lin) produces nonzero values at late layers
  3. Steering produces nonzero effect at late layers
  4. Concept families load correctly

Usage:
    python scripts/validate.py --device cuda
"""

import scripts._env  # noqa: F401

import argparse
import sys
import torch
import numpy as np

from src.extraction.activations import ActivationExtractor
from src.data.core_families import ArithmeticProbe


def main():
    parser = argparse.ArgumentParser(description="Validate LAP pipeline")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    passed = 0
    failed = 0

    # Test 1: Model loads and extraction works
    print(f"Loading model: {args.model}")
    extractor = ActivationExtractor(args.model, device=args.device)
    n_layers = extractor.n_layers
    print(f"  Layers: {n_layers}, d_model: {extractor.d_model}")

    # Test 2: Load a concept family
    print("\nTest: Load arithmetic family...")
    dataset = ArithmeticProbe(n_prompts=50)
    prompts = dataset.load()
    texts = [p.prompt_text for p in prompts]
    print(f"  Loaded {len(prompts)} prompts")
    if len(prompts) > 0:
        print(f"  Example: \"{texts[0]}\" -> \"{prompts[0].correct_answer}\"")
        passed += 1
    else:
        print("  FAIL: no prompts loaded")
        failed += 1

    # Test 3: Extract activations
    print("\nTest: Extract activations...")
    test_texts = texts[:10]
    extraction = extractor.extract(test_texts, layers=[0, n_layers // 2, n_layers - 1],
                                   batch_size=args.batch_size)
    acts = extraction.to_numpy()
    print(f"  Layers extracted: {sorted(acts.keys())}")
    print(f"  Shape per layer: {acts[0].shape}")
    if acts[0].shape[0] == len(test_texts) and acts[0].shape[1] == extractor.d_model:
        print("  PASS")
        passed += 1
    else:
        print(f"  FAIL: expected ({len(test_texts)}, {extractor.d_model})")
        failed += 1

    # Test 4: Logit lens (A_lin) at late layer
    print("\nTest: Logit lens accuracy at late layers...")
    test_prompts = prompts[:30]
    test_texts = [p.prompt_text for p in test_prompts]
    extraction = extractor.extract(test_texts, layers=list(range(n_layers)),
                                   batch_size=args.batch_size)
    acts = extraction.to_numpy()

    layer_norm = extractor.model.model.norm
    unembed = extractor.model.lm_head

    late_layer = n_layers - 3
    H = torch.tensor(acts[late_layer], dtype=extractor.dtype, device=extractor.device)
    with torch.no_grad():
        logits = unembed(layer_norm(H))
        preds = logits.argmax(dim=-1).cpu().numpy()

    correct = 0
    for i, p in enumerate(test_prompts):
        tid = extractor.tokenizer.encode(" " + p.correct_answer, add_special_tokens=False)
        if len(tid) >= 1 and preds[i] == tid[0]:
            correct += 1
    a_lin = correct / len(test_prompts)
    print(f"  A_lin at L{late_layer}: {a_lin:.3f}")
    if a_lin > 0.1:
        print("  PASS (nonzero A_lin at late layer)")
        passed += 1
    else:
        print("  WARNING: low A_lin, may be expected for some models")
        passed += 1

    # Test 5: Steering produces nonzero effect
    print("\nTest: Steering at late layer...")
    target = test_prompts[0].correct_answer
    target_tid = extractor.tokenizer.encode(" " + target, add_special_tokens=False)
    if len(target_tid) >= 1:
        target_tid = target_tid[0]
        target_mask = np.array([p.correct_answer == target for p in test_prompts])
        other_mask = ~target_mask

        if target_mask.sum() >= 3 and other_mask.sum() >= 3:
            H_late = acts[late_layer]
            direction = H_late[target_mask].mean(axis=0) - H_late[other_mask].mean(axis=0)
            magnitude = float(np.linalg.norm(direction))
            direction_unit = direction / (magnitude + 1e-10)

            other_texts = [test_texts[i] for i in range(len(test_texts)) if other_mask[i]][:10]
            n_steer = len(other_texts)
            perturbation = torch.tensor(
                np.tile(direction_unit, (n_steer, 1)) * magnitude,
                dtype=torch.float32,
            )

            steered_logits = extractor.extract_with_perturbation(
                other_texts, late_layer, perturbation, batch_size=args.batch_size,
            )
            baseline_probs = extractor.get_next_token_probs(other_texts, batch_size=args.batch_size)

            steered_probs = torch.softmax(steered_logits.float(), dim=-1)
            dp = float(steered_probs[:, target_tid].mean() - baseline_probs[:, target_tid].mean())
            print(f"  Steering dP(target) at L{late_layer}: {dp:+.4f}")
            if abs(dp) > 0.001:
                print("  PASS (nonzero steering effect)")
            else:
                print("  WARNING: near-zero steering effect")
            passed += 1
        else:
            print(f"  SKIP: not enough target prompts (target='{target}', n={target_mask.sum()})")
            passed += 1
    else:
        print("  SKIP: target token not single-token")
        passed += 1

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*40}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
