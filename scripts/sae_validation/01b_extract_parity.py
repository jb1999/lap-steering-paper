"""Extract parity prompts and split by 'odd'-target.

Parity is the regime-2 case from §4.2: A_lin=0.27 but mean-difference ΔP≈0.003.
"""
import json
import sys
sys.path.insert(0, '/mnt/hgst8t/ws/linear_acc_profile')

from src.data.probe_loader import load_all_probe_families

OUT = '/mnt/hgst8t/ws/linear_acc_profile/results/sae_validation/parity_prompts.json'

fams = load_all_probe_families(include_controlled=True)
parity = fams['c_parity']

target = 'odd'
target_prompts = [p.prompt_text for p in parity if p.candidates[p.correct_index] == target]
# Match paper protocol: no text-in-prompt filter for non-arithmetic families.
other_prompts = [p.prompt_text for p in parity
                 if p.candidates[p.correct_index] != target]

print(f'Parity total: {len(parity)}')
print(f'Target ({target}): {len(target_prompts)}')
print(f'Other (filtered): {len(other_prompts)}')
print('\nSample target:')
for p in target_prompts[:5]: print(f'  {p!r}')
print('\nSample other:')
for p in other_prompts[:5]: print(f'  {p!r}')

with open(OUT, 'w') as f:
    json.dump({
        'target': target,
        'target_prompts': target_prompts,
        'other_prompts': other_prompts,
    }, f, indent=2)
print(f'\nSaved to {OUT}')
