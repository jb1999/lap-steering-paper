"""Head-to-head: Arad et al. (2025) S_out vs our C(v_target) for SAE feature steering.

For each candidate SAE feature, compute:
  - C(v_target): our linearized score (single-row, target-conditioned, prompt-agnostic).
      v_target_row . LN(W_dec[i])  (the value used in the LAP paper)
  - S_in: Arad's input score = |T ∩ L| / |T|, where T = max-activating tokens of the
      feature across a corpus, L = top-20 logit-lens tokens. We use our existing
      target+other prompts as the corpus (deviation from Arad's Neuronpedia corpus,
      noted as caveat).
  - S_out_concept: Arad's output score, AS DEFINED. Intervene on neutral prompt
      "From my experience," with feature_acts[:, -1, i] += amp_factor * max_act_value
      (amp_factor=10, max_act_value = max activation over ALL features at last token).
      Score = (1 - min_rank_LL/V) * max_prob_LL
      where LL = top-20 logit-lens tokens of the feature.
  - S_out_target: same intervention, but score = (1 - rank(target)/V) * p(target).

Then measure ground-truth ΔP under TWO intervention regimes:
  - ΔP_ours: residual += alpha * (v_f scaled to ‖d‖), measured at the next-token position
      on target prompts (matches our existing 02_sae_steering.py protocol).
  - ΔP_arad: full SAE encode/decode + amp_factor*max_act intervention (matches Arad).

Outputs: results/sae_validation/arad_h2h_<target>_l<L>_<model>.json
"""

import json
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from scipy.stats import spearmanr

_default_cu13 = os.path.expanduser(
    '~/venvs/lap-sae/lib/python3.11/site-packages/nvidia/cu13/lib'
)
sae_lib = os.environ.get('LAP_SAE_CU13_LIB', _default_cu13)
if sae_lib and os.path.isdir(sae_lib):
    os.environ['LD_LIBRARY_PATH'] = sae_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / 'results/sae_validation'

NEUTRAL_PROMPT = "From my experience,"
ARAD_AMP_FACTOR = 10
ARAD_LL_K = 20


def get_target_token_id(tokenizer, target):
    for text in [' ' + target, target]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f'Target {target!r} does not tokenize to a single token')


def fix_position_ids(inputs):
    inputs['position_ids'] = (inputs['attention_mask'].cumsum(-1) - 1).clamp(min=0)
    return inputs


@torch.no_grad()
def extract_residuals_at_layer(model, tokenizer, prompts, layer, device,
                                batch_size=16, max_length=128):
    out_acts = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=max_length).to(device)
        fix_position_ids(inputs)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        h = outputs.hidden_states[layer + 1][:, -1, :]
        out_acts.append(h.detach().cpu())
    return torch.cat(out_acts, dim=0)


@torch.no_grad()
def measure_target_p_direct(model, tokenizer, prompts, target_tid, device,
                             steering_dir=None, alpha=1.0, layer=None,
                             batch_size=16, max_length=128):
    """ΔP measurement under our direct-add intervention regime.

    Adds alpha * steering_dir to the residual at position seq_len-1 of layer.
    Returns per-prompt P(target_tid) at the next-token position.
    """
    sd = steering_dir.to(device).float() if steering_dir is not None else None

    def make_hook(seq_len):
        def hook(module, _input, output):
            if isinstance(output, tuple):
                resid, *rest = output
                resid = resid.clone()
                resid[:, seq_len - 1, :] = resid[:, seq_len - 1, :] + alpha * sd.to(resid.dtype)
                return (resid, *rest)
            output = output.clone()
            output[:, seq_len - 1, :] = output[:, seq_len - 1, :] + alpha * sd.to(output.dtype)
            return output
        return hook

    probs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=max_length).to(device)
        fix_position_ids(inputs)
        seq_len = inputs['input_ids'].shape[1]
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


@torch.no_grad()
def measure_target_p_arad(model, tokenizer, sae, prompts, target_tid, device,
                          feature_idx, amp_factor=ARAD_AMP_FACTOR, layer=None,
                          batch_size=16, max_length=128):
    """ΔP measurement under Arad's SAE-decode intervention regime.

    On every forward pass through `model.model.layers[layer]`, intercepts the
    output residual, encodes via SAE, adds `amp_factor * max_act_value` to
    feature `feature_idx` at the last token, decodes, and adds the SAE error
    back. This matches AmlifySAEHook from the Arad et al. codebase.
    """
    def make_hook():
        def hook(module, _input, output):
            output_tensor = output[0] if isinstance(output, tuple) else output
            orig_dtype = output_tensor.dtype
            x = output_tensor.float()
            feature_acts = sae.encode(x)
            with torch.no_grad():
                # SAE error on the unmodified input
                x_recon = sae.decode(feature_acts)
                sae_error = x - x_recon
            max_act_value = feature_acts[:, -1, :].max().item()
            feature_acts[:, -1, feature_idx] = (
                feature_acts[:, -1, feature_idx] + amp_factor * max_act_value
            )
            sae_out = sae.decode(feature_acts) + sae_error
            sae_out = sae_out.to(orig_dtype)
            if isinstance(output, tuple):
                return (sae_out, *output[1:])
            return sae_out
        return hook

    probs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=max_length).to(device)
        fix_position_ids(inputs)
        handle = model.model.layers[layer].register_forward_hook(make_hook())
        try:
            outputs = model(**inputs, return_dict=True)
        finally:
            handle.remove()
        p = F.softmax(outputs.logits[:, -1, :].float(), dim=-1)[:, target_tid]
        probs.append(p.detach().cpu())
    return torch.cat(probs).numpy()


@torch.no_grad()
def compute_logit_lens_topk(W_dec, final_norm, lm_head, k=ARAD_LL_K):
    """For each feature, return top-k logit-lens tokens and their projection scores.

    Matches Arad: l(f_i) = W_unemb^T (LN(W_dec[i])) ; top-k tokens by descending logit.
    """
    n_features, _ = W_dec.shape
    W_dec_normed = final_norm(W_dec.float()).to(lm_head.weight.dtype)
    # Chunk over features to avoid materializing (n_features, V) in fp32
    chunk = 512
    all_idx = []
    all_val = []
    for i in range(0, n_features, chunk):
        logits = W_dec_normed[i:i + chunk] @ lm_head.weight.T  # (chunk, V)
        topk = torch.topk(logits, k=k, dim=-1)
        all_idx.append(topk.indices.cpu())
        all_val.append(topk.values.float().cpu())
    return torch.cat(all_idx).numpy(), torch.cat(all_val).numpy()


@torch.no_grad()
def compute_arad_output_scores(model, sae, tokenizer, neutral_prompt, layer,
                                features, ll_topk_indices, target_tid, device,
                                amp_factor=ARAD_AMP_FACTOR):
    """Run Arad's output-score intervention for each feature on the neutral prompt.

    Returns:
      s_out_concept[i]: Arad's score using the feature's own top-k LL tokens.
      s_out_target[i]:  same intervention, but rank-weighted prob on target_tid.
    """
    inputs = tokenizer(neutral_prompt, return_tensors='pt').to(device)
    fix_position_ids(inputs)

    s_out_concept = np.full(len(features), np.nan)
    s_out_target = np.full(len(features), np.nan)

    for fi, feat in enumerate(features):
        ll_idx = ll_topk_indices[feat].tolist()

        def make_hook(feature_idx):
            def hook(module, _input, output):
                output_tensor = output[0] if isinstance(output, tuple) else output
                orig_dtype = output_tensor.dtype
                x = output_tensor.float()
                feature_acts = sae.encode(x)
                with torch.no_grad():
                    x_recon = sae.decode(feature_acts)
                    sae_error = x - x_recon
                max_act_value = feature_acts[:, -1, :].max().item()
                feature_acts[:, -1, feature_idx] = (
                    feature_acts[:, -1, feature_idx] + amp_factor * max_act_value
                )
                sae_out = sae.decode(feature_acts) + sae_error
                sae_out = sae_out.to(orig_dtype)
                if isinstance(output, tuple):
                    return (sae_out, *output[1:])
                return sae_out
            return hook

        handle = model.model.layers[layer].register_forward_hook(make_hook(int(feat)))
        try:
            outputs = model(**inputs, return_dict=True)
        finally:
            handle.remove()

        logits = outputs.logits[0, -1].float().cpu()
        probs = F.softmax(logits, dim=0)
        V = probs.shape[0]

        # Concept score (Arad as-defined)
        argsort = torch.argsort(probs, descending=True)
        # Rank of each LL token under post-intervention distribution
        rank_lookup = torch.empty(V, dtype=torch.long)
        rank_lookup[argsort] = torch.arange(V)
        ll_ranks = rank_lookup[ll_idx]
        min_rank = int(ll_ranks.min().item())
        max_prob = float(probs[ll_idx].max().item())
        s_out_concept[fi] = (1.0 - min_rank / V) * max_prob

        # Target-conditioned score
        target_rank = int(rank_lookup[target_tid].item())
        target_prob = float(probs[target_tid].item())
        s_out_target[fi] = (1.0 - target_rank / V) * target_prob

    return s_out_concept, s_out_target


@torch.no_grad()
def compute_input_scores(sae, residuals, ll_topk_indices, features, tokenizer,
                          prompts, max_length=128):
    """S_in approximation: |T ∩ L| / |T|, where T is the max-activating *token*
    per prompt for each feature, collected across prompts.

    DEVIATION from Arad: they use Neuronpedia's pre-cached corpus; we use our
    own target+other prompts as the corpus (smaller, less diverse). Documented
    as a caveat. Computed feature-by-feature so we don't materialize a huge
    (n_prompts, T_max, n_features) tensor.
    """
    raise NotImplementedError(
        "S_in requires per-token activations across the corpus, which our "
        "extract_residuals_at_layer does not return. Implementing this would "
        "require re-running with full-sequence activations. Skipping for v1; "
        "the central head-to-head is C(v_target) vs S_out, both of which "
        "operate on one residual per item or no residual at all."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--prompts', required=True,
                        help='Filename in results/sae_validation/ (e.g. geography_prompts.json)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--sae-release', default=None)
    parser.add_argument('--sae-width', default='16k')
    parser.add_argument('--n-random', type=int, default=500,
                        help='Random features sampled (in addition to candidates)')
    parser.add_argument('--top-k-candidates', type=int, default=20,
                        help='Top-k by 3 criteria; union becomes the candidate set')
    parser.add_argument('--max-other', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-arad-dp', action='store_true',
                        help='Skip ΔP_arad (Arad-regime intervention measurement). '
                             'Useful for fast smoke tests.')
    parser.add_argument('--max-features', type=int, default=None,
                        help='Cap total features (random+candidate) for smoke tests')
    args = parser.parse_args()

    if args.sae_release is None:
        if 'gemma-2-2b' in args.model:
            args.sae_release = 'gemma-scope-2b-pt-res-canonical'
        elif 'gemma-2-9b' in args.model:
            args.sae_release = 'gemma-scope-9b-pt-res-canonical'
        elif 'Llama-3.1-8B' in args.model.lower() or 'llama-3.1-8b' in args.model.lower():
            args.sae_release = 'llama_scope_lxr_8x'
        else:
            raise ValueError(f'Cannot infer SAE release for {args.model!r}')

    if args.sae_release.startswith('llama_scope'):
        expansion = args.sae_release.rsplit('_', 1)[-1]
        sae_id = f'l{args.layer}r_{expansion}'
    else:
        sae_id = f'layer_{args.layer}/width_{args.sae_width}/canonical'

    PROMPTS_PATH = RESULTS / args.prompts
    OUT_PATH = RESULTS / args.out
    device = args.device
    model_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    with open(PROMPTS_PATH) as f:
        d = json.load(f)
    target = d['target']
    target_prompts = d['target_prompts']
    other_prompts = d['other_prompts'][:args.max_other]
    print(f'Target {target!r}: {len(target_prompts)} target prompts, '
          f'{len(other_prompts)} other prompts')

    print(f'\nLoading {args.model} (dtype={args.dtype})...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=model_dtype, device_map=device,
    )
    model.eval()
    target_tid = get_target_token_id(tokenizer, target)
    print(f'Target token id: {target_tid} ({tokenizer.decode([target_tid])!r})')

    print(f'\nLoading SAE: release={args.sae_release}, sae_id={sae_id}...')
    sae = SAE.from_pretrained(release=args.sae_release, sae_id=sae_id, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    W_dec = sae.W_dec.detach()
    n_features, d_model = W_dec.shape
    print(f'SAE: {n_features} features, d_model={d_model}')

    W_U = model.lm_head.weight.detach()
    final_norm = model.model.norm
    target_row = W_U[target_tid].to(device).float()

    # C(v_target) — paper convention: v . W_U[t] (no LayerNorm in our score)
    print('\nComputing C(v_target)...')
    v_proj = (W_dec.float().to(device) @ target_row).cpu().numpy()
    print(f'  C(v_target): mean={v_proj.mean():.4f}, std={v_proj.std():.4f}, '
          f'min={v_proj.min():.4f}, max={v_proj.max():.4f}')

    # Logit-lens top-k tokens per feature (Arad's "concept" set)
    print(f'\nComputing logit-lens top-{ARAD_LL_K} tokens per feature (Arad concept set)...')
    ll_topk_indices, ll_topk_values = compute_logit_lens_topk(
        W_dec.to(device), final_norm, model.lm_head, k=ARAD_LL_K
    )
    print(f'  ll_topk_indices: {ll_topk_indices.shape}')

    # Residual extraction for the activation criterion (and to scale steering)
    print(f'\nExtracting residuals at L{args.layer}...')
    target_resid = extract_residuals_at_layer(
        model, tokenizer, target_prompts, args.layer, device
    ).to(device).float()
    other_resid = extract_residuals_at_layer(
        model, tokenizer, other_prompts, args.layer, device
    ).to(device).float()
    mean_diff_dir = (target_resid.mean(dim=0) - other_resid.mean(dim=0)).cpu()
    mean_diff_norm = mean_diff_dir.norm().item()
    print(f'  mean-diff norm: {mean_diff_norm:.3f}')

    feat_acts_target = sae.encode(target_resid).detach().cpu().numpy()
    feat_acts_other = sae.encode(other_resid).detach().cpu().numpy()
    activation_score = feat_acts_target.mean(axis=0)
    activation_other = feat_acts_other.mean(axis=0)
    diff_activation = activation_score - activation_other

    # Build candidate set: union of top-K by 3 criteria + n_random uniform sample
    K = args.top_k_candidates
    top_proj = np.argsort(-v_proj)[:K]
    top_act = np.argsort(-activation_score)[:K]
    top_diff = np.argsort(-diff_activation)[:K]
    candidates_biased = sorted(set(top_proj.tolist()) | set(top_act.tolist())
                                | set(top_diff.tolist()))
    print(f'\nBiased candidate set (union of top-{K} by 3 criteria): '
          f'{len(candidates_biased)} features')

    rng = np.random.RandomState(args.seed)
    available = np.setdiff1d(np.arange(n_features), candidates_biased)
    n_rand = min(args.n_random, len(available))
    candidates_random = sorted(rng.choice(available, size=n_rand, replace=False).tolist())
    print(f'Random sample (seed={args.seed}): {len(candidates_random)} features')

    all_features = sorted(set(candidates_biased) | set(candidates_random))
    if args.max_features is not None and len(all_features) > args.max_features:
        rng2 = np.random.RandomState(args.seed + 1)
        all_features = sorted(rng2.choice(all_features,
                                          size=args.max_features, replace=False).tolist())
    print(f'Total features to evaluate: {len(all_features)}')

    is_random = {f: (f in set(candidates_random)) for f in all_features}
    is_biased = {f: (f in set(candidates_biased)) for f in all_features}

    # Baseline P(target) on other prompts (no steering)
    print(f'\nMeasuring baseline P({target!r}) on {len(other_prompts)} other prompts...')
    baseline_p = measure_target_p_direct(
        model, tokenizer, other_prompts, target_tid, device
    )
    print(f'  baseline = {baseline_p.mean():.5f}')

    # Mean-difference baseline (sanity)
    md_steered_p = measure_target_p_direct(
        model, tokenizer, other_prompts, target_tid, device,
        steering_dir=mean_diff_dir, alpha=1.0, layer=args.layer,
    )
    md_dp = float((md_steered_p - baseline_p).mean())
    print(f'  mean-diff ΔP = {md_dp:+.4f}')

    # Compute Arad output scores (concept + target) on the neutral prompt
    print(f'\nComputing Arad output scores on neutral prompt {NEUTRAL_PROMPT!r}...')
    s_out_concept, s_out_target = compute_arad_output_scores(
        model, sae, tokenizer, NEUTRAL_PROMPT, args.layer,
        all_features, ll_topk_indices, target_tid, device,
    )
    print(f'  S_out_concept: mean={np.nanmean(s_out_concept):.4f}, '
          f'max={np.nanmax(s_out_concept):.4f}')
    print(f'  S_out_target:  mean={np.nanmean(s_out_target):.4f}, '
          f'max={np.nanmax(s_out_target):.4f}')

    # Measure ΔP under both intervention regimes
    print(f'\nMeasuring ΔP under both regimes for {len(all_features)} features...')
    feature_results = []
    for fi, f in enumerate(all_features):
        v_f = W_dec[f].cpu().float()
        scale = mean_diff_norm / max(v_f.norm().item(), 1e-6)
        v_f_scaled = v_f * scale
        steered_p_ours = measure_target_p_direct(
            model, tokenizer, other_prompts, target_tid, device,
            steering_dir=v_f_scaled, alpha=1.0, layer=args.layer,
        )
        dp_ours = float((steered_p_ours - baseline_p).mean())

        if args.skip_arad_dp:
            dp_arad = None
        else:
            steered_p_arad = measure_target_p_arad(
                model, tokenizer, sae, other_prompts, target_tid, device,
                feature_idx=int(f), layer=args.layer,
            )
            dp_arad = float((steered_p_arad - baseline_p).mean())

        feature_results.append({
            'feature_id': int(f),
            'is_biased_candidate': is_biased[f],
            'is_random_sample': is_random[f],
            'C_v_target': float(v_proj[f]),
            'activation_target': float(activation_score[f]),
            'activation_other': float(activation_other[f]),
            'diff_activation': float(diff_activation[f]),
            'feature_norm': float(v_f.norm().item()),
            'S_out_concept': float(s_out_concept[fi]),
            'S_out_target': float(s_out_target[fi]),
            'delta_p_ours': dp_ours,
            'delta_p_arad': dp_arad,
            'll_top5_token_ids': ll_topk_indices[f, :5].tolist(),
        })
        if (fi + 1) % 20 == 0 or fi == len(all_features) - 1:
            print(f'  [{fi+1}/{len(all_features)}] feat {f}: '
                  f'C(v)={v_proj[f]:+.3f}, S_out_c={s_out_concept[fi]:.3f}, '
                  f'S_out_t={s_out_target[fi]:.3f}, '
                  f'dp_ours={dp_ours:+.4f}'
                  + (f', dp_arad={dp_arad:+.4f}' if dp_arad is not None else ''))

    # Headline correlations
    fr = feature_results
    arr = lambda k: np.array([f[k] for f in fr if f[k] is not None])
    arr_pair = lambda kx, ky: (
        np.array([f[kx] for f in fr if f[ky] is not None]),
        np.array([f[ky] for f in fr if f[ky] is not None]),
    )

    correlations = {}
    score_keys = ['C_v_target', 'S_out_concept', 'S_out_target',
                  'activation_target', 'diff_activation']
    dp_keys = ['delta_p_ours'] + ([] if args.skip_arad_dp else ['delta_p_arad'])

    print(f'\n=== Correlations across {len(fr)} features ===')
    for dpk in dp_keys:
        for sk in score_keys:
            x, y = arr_pair(sk, dpk)
            if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
                rho, p = float('nan'), float('nan')
            else:
                rho, p = spearmanr(x, y)
            correlations[f'{sk}_vs_{dpk}'] = {'rho': float(rho), 'p': float(p),
                                                'n': int(len(x))}
            print(f'  ρ({sk:18s}, {dpk:14s}) = {rho:+.3f}  (p={p:.2e}, n={len(x)})')

    # Top-k filtering performance: take top-N by each score, mean ΔP among them
    print(f'\n=== Top-N filtering by each score (ΔP_ours) ===')
    topk_filt = {}
    for sk in score_keys:
        for n in [5, 10, 20]:
            xs = np.array([f[sk] for f in fr])
            ys_ours = np.array([f['delta_p_ours'] for f in fr])
            order = np.argsort(-xs)[:n]
            mean_dp = float(np.nanmean(ys_ours[order]))
            topk_filt[f'{sk}_top{n}_dp_ours'] = mean_dp
            print(f'  top-{n:2d} by {sk:18s} -> mean ΔP_ours = {mean_dp:+.4f}')

    out = {
        'config': vars(args),
        'target': target,
        'target_tid': target_tid,
        'neutral_prompt': NEUTRAL_PROMPT,
        'arad_amp_factor': ARAD_AMP_FACTOR,
        'arad_ll_k': ARAD_LL_K,
        'baseline_p_mean': float(baseline_p.mean()),
        'mean_diff_dp': md_dp,
        'mean_diff_norm': float(mean_diff_norm),
        'n_biased_candidates': len(candidates_biased),
        'n_random_sample': len(candidates_random),
        'feature_results': feature_results,
        'correlations': correlations,
        'topk_filtering': topk_filt,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved results to {OUT_PATH}')


if __name__ == '__main__':
    main()
