"""Extract geography prompts and split by Spanish-target.

Saves prompts + the steering target to results/sae_validation/geography_prompts.json
so the SAE-venv script can load them without depending on src/data/.
"""
import json
import sys
sys.path.insert(0, '/mnt/hgst8t/ws/linear_acc_profile')

from src.data.probe_loader import load_all_probe_families

OUT = '/mnt/hgst8t/ws/linear_acc_profile/results/sae_validation/geography_prompts.json'

fams = load_all_probe_families()
geo = fams['geography']

target = 'Spanish'
target_prompts = [p.prompt_text for p in geo if p.candidates[p.correct_index] == target]
# Match the paper's run_cross_concept.py protocol: for non-arithmetic families,
# other_mask = ~target_mask (no text-in-prompt filter).
other_prompts = [p.prompt_text for p in geo
                 if p.candidates[p.correct_index] != target]

print(f'Geography total: {len(geo)}')
print(f'Target ({target}): {len(target_prompts)} prompts')
print(f'Other (filtered): {len(other_prompts)} prompts')
print()
print('Sample target prompts:')
for p in target_prompts[:5]:
    print(f'  {p!r}')
print('Sample other prompts:')
for p in other_prompts[:5]:
    print(f'  {p!r}')

with open(OUT, 'w') as f:
    json.dump({
        'target': target,
        'target_prompts': target_prompts,
        'other_prompts': other_prompts,
    }, f, indent=2)
print(f'\nSaved to {OUT}')
