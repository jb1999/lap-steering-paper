"""Experiment 3: Failure Mode Clustering (H4)

For prompts where the model fails (incorrect top-1 prediction),
cluster by per-prompt LAP features and test whether clusters
correspond to distinct failure types.

Per-prompt features available:
- λ (perturbation sensitivity) at each layer
- Projection onto concept direction (how aligned is this prompt's
  activation with the steering direction?)
- Norm of activation (how "loud" is the representation?)
- Model's P(correct) (how confident was it, even though wrong?)
- Distance from centroid (how typical is this prompt?)

Usage:
    python scripts/run_experiment3.py --results-dir results
"""

import scripts._env  # noqa: F401 — must be first

import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def run_experiment3(args):
    results_dir = Path(args.results_dir) / "exp3"
    results_dir.mkdir(parents=True, exist_ok=True)
    exp1_dir = Path(args.results_dir) / "exp1"
    exp2_dir = Path(args.results_dir) / "exp2"

    # Load families
    from src.data.loader import load_families
    families = load_families(n_prompts=args.n_prompts)

    # Load steering targets for concept directions
    with open(exp2_dir / "steering_results.json") as f:
        steer = json.load(f)

    # Load model accuracy info
    with open(exp1_dir / "layer_accuracy.json") as f:
        exp1_lin = json.load(f)

    # Pick the best layer for analysis (where linear acc is highest on average)
    avg_by_layer = {}
    for family in exp1_lin:
        for l, data in exp1_lin[family]["layers"].items():
            avg_by_layer.setdefault(l, []).append(data["linear_acc"])
    best_layer = max(avg_by_layer.keys(), key=lambda l: np.mean(avg_by_layer[l]))
    print(f"Analysis layer: L{best_layer} (highest average linear accuracy)")

    all_results = {}

    for family_name, prompts in families.items():
        print(f"\n{'=' * 60}")
        print(f"Failure Clustering: {family_name}")
        print("=" * 60)

        # Load cached activations and perturbation sensitivity
        gpu_cache = exp1_dir / "gpu_cache" / f"{family_name}_gpu.npz"
        if not gpu_cache.exists():
            print(f"  No GPU cache, skipping")
            continue

        cached = np.load(gpu_cache)
        H = cached[f"act_{best_layer}"]  # (n_prompts, d_model)
        lambda_vals = cached[f"ps_{best_layer}"]  # (n_prompts,)

        # Get correct token IDs and model predictions
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model
        )

        def get_tid(word):
            for text in [" " + word, word]:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) == 1:
                    return ids[0]
            return tokenizer.encode(word, add_special_tokens=False)[0]

        correct_tids = np.array([get_tid(p.correct_answer) for p in prompts])

        # Determine which prompts model gets right/wrong
        # Check for cached correct_mask
        mask_file = results_dir / f"{family_name}_correct_mask.npy"
        if mask_file.exists():
            correct_mask = np.load(mask_file)
        else:
            # Need model to evaluate — lazy load once
            if 'extractor' not in dir():
                print("  Loading model to evaluate correct/incorrect...")
                from src.extraction.activations import ActivationExtractor
                import torch, gc, string
                extractor = ActivationExtractor(
                    args.model, device=args.device
                )
                vocab_size = extractor.tokenizer.vocab_size
                punct_and_space = set(string.punctuation + string.whitespace)
                content_mask = torch.ones(vocab_size, dtype=torch.bool)
                for tid in range(vocab_size):
                    decoded = extractor.tokenizer.decode([tid])
                    if not decoded or all(c in punct_and_space for c in decoded):
                        content_mask[tid] = False

            prompt_texts = [p.prompt_text for p in prompts]
            probs = extractor.get_next_token_probs(prompt_texts, batch_size=32)
            content_probs = probs.clone()
            content_probs[:, ~content_mask] = 0

            correct_mask = np.zeros(len(prompts), dtype=bool)
            for i in range(len(prompts)):
                pred_tid = content_probs[i].argmax().item()
                true_tid = get_tid(prompts[i].correct_answer)
                correct_mask[i] = (pred_tid == true_tid)

            np.save(mask_file, correct_mask)
            print(f"  Saved correct mask: {correct_mask.sum()}/{len(prompts)} correct")

        n_correct = correct_mask.sum()
        n_incorrect = (~correct_mask).sum()
        print(f"  Correct: {n_correct}, Incorrect: {n_incorrect}")

        if n_incorrect < 10:
            print(f"  Too few failures to cluster")
            all_results[family_name] = {"n_incorrect": int(n_incorrect),
                                         "status": "too_few_failures"}
            continue

        # Compute per-prompt features for failed prompts
        H_fail = H[~correct_mask]
        H_correct = H[correct_mask]
        lambda_fail = lambda_vals[~correct_mask]
        lambda_correct = lambda_vals[correct_mask]

        # Feature 1: Perturbation sensitivity (λ)
        f_lambda = lambda_fail

        # Feature 2: Activation norm
        f_norm = np.linalg.norm(H_fail, axis=1)

        # Feature 3: Distance from centroid (Mahalanobis-like)
        centroid = H.mean(axis=0)
        f_dist = np.linalg.norm(H_fail - centroid, axis=1)

        # Feature 4: Projection onto concept direction
        target_answer = steer.get(family_name, {}).get("target", prompts[0].correct_answer)
        target_mask = np.array([p.correct_answer == target_answer for p in prompts])
        if target_mask.sum() > 0 and (~target_mask).sum() > 0:
            direction = H[target_mask].mean(axis=0) - H[~target_mask].mean(axis=0)
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-10:
                direction = direction / dir_norm
                f_proj = H_fail @ direction
            else:
                f_proj = np.zeros(len(H_fail))
        else:
            f_proj = np.zeros(len(H_fail))

        # Feature 5: Cosine similarity to correct-prompt centroid
        correct_centroid = H_correct.mean(axis=0)
        correct_centroid_norm = np.linalg.norm(correct_centroid)
        if correct_centroid_norm > 1e-10:
            f_cos_correct = (H_fail @ correct_centroid) / (
                np.linalg.norm(H_fail, axis=1) * correct_centroid_norm
            )
        else:
            f_cos_correct = np.zeros(len(H_fail))

        # Stack features
        feature_names = ["lambda", "act_norm", "dist_from_centroid",
                        "concept_projection", "cos_to_correct"]
        X_fail = np.column_stack([f_lambda, f_norm, f_dist, f_proj, f_cos_correct])

        # Remove NaN/inf
        valid = np.all(np.isfinite(X_fail), axis=1)
        X_fail_clean = X_fail[valid]
        print(f"  Valid failed prompts: {valid.sum()}/{len(X_fail)}")

        if valid.sum() < 10:
            print(f"  Too few valid failures")
            all_results[family_name] = {"n_incorrect": int(n_incorrect),
                                         "status": "too_few_valid"}
            continue

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_fail_clean)

        # Find best k by silhouette score
        k_range = range(2, min(6, len(X_scaled) // 5 + 1))
        silhouettes = {}
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                silhouettes[k] = silhouette_score(X_scaled, labels)

        if not silhouettes:
            print("  Could not cluster")
            all_results[family_name] = {"n_incorrect": int(n_incorrect),
                                         "status": "clustering_failed"}
            continue

        best_k = max(silhouettes, key=silhouettes.get)
        best_sil = silhouettes[best_k]
        print(f"  Best k={best_k}, silhouette={best_sil:.3f}")
        print(f"  Silhouettes: {silhouettes}")

        # Final clustering
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        # PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)

        # Analyze clusters
        overall_mean = X_scaled.mean(axis=0)
        clusters = []
        for c in range(best_k):
            mask = labels == c
            centroid_scaled = X_scaled[mask].mean(axis=0)
            centroid_raw = X_fail_clean[mask].mean(axis=0)

            # Which features deviate most from overall mean?
            deviations = centroid_scaled - overall_mean
            top_feat_idx = np.argsort(np.abs(deviations))[::-1]

            # Interpret the cluster
            dominant = feature_names[top_feat_idx[0]]
            sign = "high" if deviations[top_feat_idx[0]] > 0 else "low"

            # Map to failure taxonomy
            if dominant == "lambda" and sign == "high":
                taxonomy = "chaotic_regime"
            elif dominant == "dist_from_centroid" and sign == "high":
                taxonomy = "distribution_edge"
            elif dominant == "concept_projection" and sign == "low":
                taxonomy = "wrong_direction"
            elif dominant == "cos_to_correct" and sign == "low":
                taxonomy = "unlike_correct"
            elif dominant == "act_norm" and sign == "low":
                taxonomy = "weak_activation"
            elif dominant == "act_norm" and sign == "high":
                taxonomy = "noisy_activation"
            else:
                taxonomy = f"{sign}_{dominant}"

            cluster_info = {
                "id": c,
                "n_prompts": int(mask.sum()),
                "taxonomy": taxonomy,
                "dominant_feature": dominant,
                "dominant_sign": sign,
                "feature_means": {fn: float(centroid_raw[i])
                                  for i, fn in enumerate(feature_names)},
                "feature_deviations": {fn: float(deviations[i])
                                       for i, fn in enumerate(feature_names)},
            }
            clusters.append(cluster_info)

            print(f"\n  Cluster {c}: n={mask.sum()}, type={taxonomy}")
            print(f"    Dominant: {dominant} ({sign})")
            for i, fn in enumerate(feature_names):
                dev = deviations[i]
                bar = "+" * int(max(0, dev) * 5) + "-" * int(max(0, -dev) * 5)
                print(f"    {fn:<25} mean={centroid_raw[i]:>10.2f} dev={dev:>+6.2f} {bar}")

        # Compare cluster features with correct prompts
        correct_features = np.column_stack([
            lambda_correct, np.linalg.norm(H_correct, axis=1),
            np.linalg.norm(H_correct - centroid, axis=1),
            H_correct @ direction if 'direction' in dir() and direction is not None else np.zeros(len(H_correct)),
            (H_correct @ correct_centroid) / (
                np.linalg.norm(H_correct, axis=1) * correct_centroid_norm + 1e-10
            ),
        ])

        print(f"\n  Correct prompts (reference):")
        for i, fn in enumerate(feature_names):
            print(f"    {fn:<25} mean={correct_features[:, i].mean():>10.2f}")

        all_results[family_name] = {
            "n_incorrect": int(n_incorrect),
            "n_correct": int(n_correct),
            "best_k": best_k,
            "silhouette": best_sil,
            "silhouettes_by_k": {str(k): v for k, v in silhouettes.items()},
            "clusters": clusters,
            "analysis_layer": int(best_layer),
            "correct_feature_means": {fn: float(correct_features[:, i].mean())
                                      for i, fn in enumerate(feature_names)},
        }

    # Free model if loaded
    if 'extractor' in dir():
        del extractor
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    # Save
    with open(results_dir / "clustering_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'=' * 60}")
    print("FAILURE CLUSTERING SUMMARY")
    print("=" * 60)

    print(f"\n{'Family':<16} {'#Fail':>6} {'k':>3} {'Sil':>6} {'Cluster Types'}")
    print("-" * 70)
    for family, result in all_results.items():
        if "clusters" not in result:
            print(f"{family:<16} {result.get('n_incorrect', 0):>6} — {result.get('status', 'N/A')}")
            continue
        types = ", ".join(f"{c['taxonomy']}({c['n_prompts']})" for c in result["clusters"])
        print(f"{family:<16} {result['n_incorrect']:>6} {result['best_k']:>3} "
              f"{result['silhouette']:>6.3f} {types}")

    print(f"\nExperiment 3 complete. Results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Failure Clustering")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--n-prompts", type=int, default=2000)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_experiment3(args)


if __name__ == "__main__":
    main()
