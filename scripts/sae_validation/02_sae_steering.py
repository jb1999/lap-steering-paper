"""SAE feature steering validation, matching the paper's run_steerability.py protocol.

Protocol (mirrors ~/adata/lap-steering-paper/scripts/run_steerability.py):
  - Model loaded in bfloat16
  - Tokenizer left-padded
  - Extraction: residuals from out.hidden_states[l+1][:, -1, :] WITHOUT position_ids fix
    (matches the paper's mean-difference direction)
  - Baseline + steered measurement: position_ids fix applied
    (inp["position_ids"] = (inp["attention_mask"].cumsum(-1) - 1).clamp(min=0))
  - Injection at position seq_len - 1 (= true last real token under left-padding)
  - Mean-difference direction at its natural norm; alpha = magnitude * 1.0

For each of the 16K GemmaScope residual SAE features at the target layer:
  - Compute decoder direction v_f (one row of W_dec).
  - Compute C(v_f)_target = v_f^T W_U[target] (the linearized contribution to the
    target logit). This is the SAE-feature analogue of A_lin.
  - Compute mean activation a_f on target prompts at the layer.

Then steer with each top-K feature individually and with multi-feature sums of
top-k by each criterion (k in {1, 5, 20, 100}). All directions scaled to the
mean-difference norm so the comparison isolates *direction* from *magnitude*.

Outputs: results/sae_validation/<out>.json
"""

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from scipy.stats import spearmanr

# CUDA 13 libs in venv: torch wheels for cu13 ship .so files inside the venv
# under nvidia/cu13/lib; some hosts need this path on LD_LIBRARY_PATH for the
# loader to find them. Override with LAP_SAE_CU13_LIB=/path/to/cu13/lib if your
# venv lives elsewhere; set LAP_SAE_CU13_LIB="" to disable.
_default_cu13 = os.path.expanduser(
    '~/venvs/lap-sae/lib/python3.11/site-packages/nvidia/cu13/lib'
)
sae_lib = os.environ.get('LAP_SAE_CU13_LIB', _default_cu13)
if sae_lib and os.path.isdir(sae_lib):
    os.environ['LD_LIBRARY_PATH'] = sae_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / 'results/sae_validation'


def get_target_token_id(tokenizer, target):
    """Match Gemma convention: leading-space token preferred."""
    for text in [' ' + target, target]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f'Target {target!r} does not tokenize to a single token')


def fix_position_ids(inputs):
    """Apply the paper's position_ids fix for left-padded RoPE inputs."""
    inputs['position_ids'] = (inputs['attention_mask'].cumsum(-1) - 1).clamp(min=0)
    return inputs


@torch.no_grad()
def extract_residuals_at_layer(model, tokenizer, prompts, layer, device,
                                batch_size=16, max_length=128, fix_pos=True):
    """Extract residual at the last (rightmost) position from out.hidden_states[layer+1].

    With fix_pos=True (default, principled), position_ids are corrected for
    left-padded inputs so that real tokens occupy positions 0..n_real-1, matching
    normal un-padded Gemma inference. This is what GemmaScope SAEs were trained on.

    With fix_pos=False, no fix -- this matches the paper's run_steerability.py
    extraction step (which is internally inconsistent with its measurement step,
    but it is the protocol that produced the published baseline numbers).
    """
    out_acts = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                           truncation=True, max_length=max_length).to(device)
        if fix_pos:
            fix_position_ids(inputs)
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        h = outputs.hidden_states[layer + 1][:, -1, :]  # (B, d)
        out_acts.append(h.detach().cpu())
    return torch.cat(out_acts, dim=0)


@torch.no_grad()
def measure_target_p(model, tokenizer, prompts, target_tid, device,
                     steering_dir=None, alpha=1.0, layer=None,
                     batch_size=16, max_length=128):
    """Return P(target_tid) per prompt at the next-token position.

    Steering: if steering_dir is provided, add alpha * steering_dir at position
    seq_len-1 (the true last real token under left-padding).

    Position_ids fix is applied (matches the paper's measurement protocol).
    """
    sd = steering_dir.to(device) if steering_dir is not None else None

    def make_hook(seq_len):
        # Inject at the rightmost position (seq_len - 1 == -1 under left-padding)
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
        logits = outputs.logits  # (B, T, V); rightmost position = next-token prediction
        # softmax in float32 to match paper's `out.logits[:, -1, :].float()`
        p = F.softmax(logits[:, -1, :].float(), dim=-1)[:, target_tid]
        probs.append(p.detach().cpu())
    return torch.cat(probs).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='google/gemma-2-2b')
    parser.add_argument('--layer', type=int, default=22)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Top features per criterion to test single-feature steering')
    parser.add_argument('--multi-ks', type=int, nargs='+', default=[1, 5, 20, 100],
                        help='Multi-feature top-k values')
    parser.add_argument('--max-other', type=int, default=200,
                        help='Cap non-target prompts (matches paper: [:200])')
    parser.add_argument('--prompts', default='geography_prompts.json')
    parser.add_argument('--out', default='sae_geography_l22.json')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16',
                        help='Model dtype (paper uses bfloat16)')
    parser.add_argument('--paper-protocol', action='store_true',
                        help='Skip the position_ids fix at extraction (matches paper).')
    parser.add_argument('--sae-release', default=None,
                        help='SAE release name. Defaults to gemma-scope-{2b,9b}-pt-res-canonical '
                             'inferred from --model.')
    parser.add_argument('--sae-width', default='16k',
                        help='SAE width (default 16k).')
    args = parser.parse_args()

    # SAE release naming differs across families (GemmaScope vs Llama-Scope).
    # GemmaScope: release = 'gemma-scope-{2b,9b}-pt-res-canonical', sae_id = 'layer_{L}/width_{W}/canonical'
    # Llama-Scope: release = 'llama_scope_lxr_{8x,32x}', sae_id = 'l{L}r_{8x,32x}'
    sae_id_template = None  # set per-release below; consumed where sae_id is constructed
    if args.sae_release is None:
        if 'gemma-2-2b' in args.model:
            args.sae_release = 'gemma-scope-2b-pt-res-canonical'
        elif 'gemma-2-9b' in args.model:
            args.sae_release = 'gemma-scope-9b-pt-res-canonical'
        elif 'Llama-3.1-8B' in args.model or 'llama-3.1-8b' in args.model.lower():
            args.sae_release = 'llama_scope_lxr_8x'
        else:
            raise ValueError(f'Cannot infer SAE release for model {args.model!r}; '
                             f'pass --sae-release explicitly.')
    if args.sae_release.startswith('llama_scope'):
        # Llama-Scope sae_id format: l{layer}r_{expansion}
        expansion = args.sae_release.rsplit('_', 1)[-1]  # '8x' or '32x'
        sae_id_template = f'l{{layer}}r_{expansion}'
    else:
        # GemmaScope canonical sae_id format: layer_{layer}/width_{width}/canonical
        sae_id_template = f'layer_{{layer}}/width_{args.sae_width}/canonical'

    PROMPTS_PATH = RESULTS / args.prompts
    OUT_PATH = RESULTS / args.out
    device = args.device
    model_dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32

    # Load prompts
    with open(PROMPTS_PATH) as f:
        d = json.load(f)
    target = d['target']
    target_prompts = d['target_prompts']
    other_prompts = d['other_prompts'][:args.max_other]
    print(f'Target {target!r}: {len(target_prompts)} target prompts')
    print(f'Other (capped at {args.max_other}): {len(other_prompts)} prompts')

    # Load model
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

    # Load SAE (sae_id template was set above based on release family)
    sae_id = sae_id_template.format(layer=args.layer)
    print(f'\nLoading SAE: release={args.sae_release}, sae_id={sae_id}...')
    sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=sae_id,
        device=device,
    )
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    W_dec = sae.W_dec.detach()  # (n_features, d_model)
    n_features, d_model = W_dec.shape
    print(f'SAE: {n_features} features, d_model={d_model}')

    # Per-feature scores
    W_U = model.lm_head.weight.detach()  # (V, d_model)
    target_row = W_U[target_tid].to(device).float()
    print(f'Target unembedding row norm: {target_row.norm().item():.3f}')

    print('\nComputing per-feature C(v_f)_target...')
    v_proj = (W_dec.float().to(device) @ target_row).cpu().numpy()
    print(f'  C(v)_target: mean={v_proj.mean():.4f}, std={v_proj.std():.4f}, '
          f'min={v_proj.min():.4f}, max={v_proj.max():.4f}')

    # Extract residuals
    fix_pos = not args.paper_protocol
    mode = 'paper-inconsistent (no fix)' if not fix_pos else 'principled (position_ids fix)'
    print(f'\nExtracting residuals at L{args.layer} ({mode})...')
    target_resid = extract_residuals_at_layer(model, tokenizer, target_prompts,
                                              args.layer, device, fix_pos=fix_pos).to(device).float()
    other_resid = extract_residuals_at_layer(model, tokenizer, other_prompts,
                                             args.layer, device, fix_pos=fix_pos).to(device).float()
    print(f'  target residual: {target_resid.shape}, other residual: {other_resid.shape}')

    # Mean-difference direction (paper protocol)
    mean_target = target_resid.mean(dim=0)
    mean_other = other_resid.mean(dim=0)
    mean_diff_dir = (mean_target - mean_other).cpu()  # (d,)
    mean_diff_norm = mean_diff_dir.norm().item()
    print(f'\nMean-difference direction norm: {mean_diff_norm:.3f}')

    # SAE feature activations on target prompts (for the activation criterion)
    feat_acts_target = sae.encode(target_resid).detach().cpu().numpy()
    feat_acts_other = sae.encode(other_resid).detach().cpu().numpy()
    activation_score = feat_acts_target.mean(axis=0)
    activation_other = feat_acts_other.mean(axis=0)
    diff_activation = activation_score - activation_other
    print(f'  activation score: max={activation_score.max():.3f}, '
          f'fraction>0 = {(activation_score>0).mean():.3f}')

    # Baseline P(target) on other prompts (no steering)
    baseline_p = measure_target_p(model, tokenizer, other_prompts, target_tid, device)
    print(f'\nBaseline P({target!r}) = {baseline_p.mean():.5f}')

    # Mean-difference baseline: inject mean_diff_dir at L22 with alpha=1.0
    # Note: paper uses alpha = magnitude * 1.0, i.e., injects the FULL direction
    # (not the unit-normalized direction). We do the same.
    md_steered_p = measure_target_p(
        model, tokenizer, other_prompts, target_tid, device,
        steering_dir=mean_diff_dir, alpha=1.0, layer=args.layer,
    )
    md_dp = (md_steered_p - baseline_p).mean()
    print(f'Mean-difference ΔP = {md_dp:.5f}  '
          f'(baseline {baseline_p.mean():.5f} → steered {md_steered_p.mean():.5f})')

    # Single-feature steering
    K = args.top_k
    top_proj = np.argsort(-v_proj)[:K]
    top_act = np.argsort(-activation_score)[:K]
    top_diff = np.argsort(-diff_activation)[:K]
    candidate = sorted(set(top_proj.tolist()) | set(top_act.tolist())
                       | set(top_diff.tolist()))
    print(f'\nSingle-feature steering: {len(candidate)} candidate features '
          f'(union of top-{K} by 3 criteria)')

    feature_results = []
    for fi, f in enumerate(candidate):
        v_f = W_dec[f].cpu().float()
        # Scale to mean-diff norm so comparison is on direction, not magnitude
        scale = mean_diff_norm / max(v_f.norm().item(), 1e-6)
        v_f_scaled = v_f * scale
        steered_p = measure_target_p(
            model, tokenizer, other_prompts, target_tid, device,
            steering_dir=v_f_scaled, alpha=1.0, layer=args.layer,
        )
        dp = (steered_p - baseline_p).mean()
        feature_results.append({
            'feature_id': int(f),
            'C_v_target': float(v_proj[f]),
            'activation_target': float(activation_score[f]),
            'activation_other': float(activation_other[f]),
            'diff_activation': float(diff_activation[f]),
            'feature_norm': float(v_f.norm().item()),
            'delta_p': float(dp),
        })
        if (fi + 1) % 10 == 0 or fi == len(candidate) - 1:
            print(f'  [{fi+1}/{len(candidate)}] feat {f}: '
                  f'C(v)_target={v_proj[f]:+.3f}, act={activation_score[f]:.3f}, '
                  f'diff_act={diff_activation[f]:+.3f}, dp={dp:+.4f}')

    # Multi-feature steering by C(v)_target
    print('\nMulti-feature steering by C(v)_target ranking:')
    multi_proj = {}
    for k in args.multi_ks:
        idx = np.argsort(-v_proj)[:k]
        v_multi = W_dec[idx].cpu().float().sum(dim=0)
        scale = mean_diff_norm / max(v_multi.norm().item(), 1e-6)
        steered_p = measure_target_p(
            model, tokenizer, other_prompts, target_tid, device,
            steering_dir=v_multi * scale, alpha=1.0, layer=args.layer,
        )
        dp = (steered_p - baseline_p).mean()
        multi_proj[f'k={k}'] = {'k': k, 'feature_ids': idx.tolist(), 'delta_p': float(dp)}
        print(f'  k={k:>3}: ΔP = {dp:+.4f}')

    # Multi-feature steering by activation
    print('\nMulti-feature steering by activation ranking:')
    multi_act = {}
    for k in args.multi_ks:
        idx = np.argsort(-activation_score)[:k]
        v_multi = W_dec[idx].cpu().float().sum(dim=0)
        scale = mean_diff_norm / max(v_multi.norm().item(), 1e-6)
        steered_p = measure_target_p(
            model, tokenizer, other_prompts, target_tid, device,
            steering_dir=v_multi * scale, alpha=1.0, layer=args.layer,
        )
        dp = (steered_p - baseline_p).mean()
        multi_act[f'k={k}'] = {'k': k, 'feature_ids': idx.tolist(), 'delta_p': float(dp)}
        print(f'  k={k:>3}: ΔP = {dp:+.4f}')

    # Correlations
    fr = feature_results
    proj_arr = np.array([f['C_v_target'] for f in fr])
    act_arr = np.array([f['activation_target'] for f in fr])
    diff_arr = np.array([f['diff_activation'] for f in fr])
    dp_arr = np.array([f['delta_p'] for f in fr])

    rho_proj, p_proj = spearmanr(proj_arr, dp_arr)
    rho_act, p_act = spearmanr(act_arr, dp_arr)
    rho_diff, p_diff = spearmanr(diff_arr, dp_arr)
    print(f'\n=== Correlations across {len(fr)} candidate features ===')
    print(f'ρ(C(v)_target, ΔP)        = {rho_proj:+.3f} (p={p_proj:.2e})')
    print(f'ρ(activation, ΔP)         = {rho_act:+.3f} (p={p_act:.2e})')
    print(f'ρ(diff_activation, ΔP)    = {rho_diff:+.3f} (p={p_diff:.2e})')

    out = {
        'config': vars(args),
        'target': target,
        'target_tid': target_tid,
        'n_target_prompts': len(target_prompts),
        'n_other_prompts': len(other_prompts),
        'baseline_p_mean': float(baseline_p.mean()),
        'mean_diff': {
            'direction_norm': float(mean_diff_norm),
            'alpha': 1.0,
            'delta_p': float(md_dp),
            'steered_p_mean': float(md_steered_p.mean()),
        },
        'feature_results': feature_results,
        'multi_feature_by_C_v_target': multi_proj,
        'multi_feature_by_activation': multi_act,
        'correlations': {
            'C_v_target_vs_dp': {'rho': float(rho_proj), 'p': float(p_proj)},
            'activation_vs_dp': {'rho': float(rho_act), 'p': float(p_act)},
            'diff_activation_vs_dp': {'rho': float(rho_diff), 'p': float(p_diff)},
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved results to {OUT_PATH}')


if __name__ == '__main__':
    main()
