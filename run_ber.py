import sys
import os
import csv
import torch

sys.path.append('/home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI')

from config import (
    NETWORK_NAME, DATASET, ROOT, DATASET_ROOT, SEED, BATCH_SIZE, DEVICE,
    P_VALUES, PILOT_TRIALS, MAX_TRIALS, PRECISION_E, CONFIDENCE_T
)
from utils import get_network, get_loader
from FaultGenerators.WeightFaultInjector import WeightFaultInjector
from BERCampaign import BERCampaign

# ----------------------------------------------------------------
# Setup
# ----------------------------------------------------------------
device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

print(f'Device:  {device}')
print(f'Network: {NETWORK_NAME}')
print(f'Dataset: {DATASET}')

network = get_network(network_name=NETWORK_NAME, device=device, dataset_name=DATASET, root=ROOT)
_, loader = get_loader(network_name=NETWORK_NAME, batch_size=BATCH_SIZE, dataset_name=DATASET, root=DATASET_ROOT)
injector = WeightFaultInjector(network)

# ----------------------------------------------------------------
# Compute injection levels from P_VALUES
# ----------------------------------------------------------------
_tmp = BERCampaign(
    network=network, loader=loader, injector=injector, device=device,
    injection_levels=[1], sampling_mode='constant',
    network_name=NETWORK_NAME, dataset_name=DATASET, root=ROOT, seed=SEED
)
total_bits = _tmp.total_bits

injection_levels = [max(1, round(p * total_bits)) for p in P_VALUES]

print(f'\nTotal bits: {total_bits}')
print('Injection levels grid:')
for p, k in zip(P_VALUES, injection_levels):
    print(f'  p={p:.0e}  →  k={k} bits')


# ----------------------------------------------------------------
# Run campaign
# ----------------------------------------------------------------
campaign = BERCampaign(
    network=network,
    loader=loader,
    injector=injector,
    device=device,
    injection_levels=injection_levels,
    sampling_mode='constant',
    network_name=NETWORK_NAME,
    dataset_name=DATASET,
    root=ROOT,
    pilot_trials=PILOT_TRIALS,
    max_trials=MAX_TRIALS,
    precision_e=PRECISION_E,
    confidence_t=CONFIDENCE_T,
    seed=SEED,
)

results = campaign.run()

# ----------------------------------------------------------------
# Save results to CSV
# ----------------------------------------------------------------
output_dir = os.path.join(ROOT, 'results', 'ber_results', DATASET, NETWORK_NAME)
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, f'{NETWORK_NAME}_{DATASET}_ber.csv')

fieldnames = [
    'network', 'dataset', 'p_value', 'injection_level',
    'golden_accuracy', 'mean_masking_ratio', 'std_masking_ratio',
    'mean_critical_ratio', 'std_critical_ratio', 'mean_faulty_accuracy',
    'n_trials', 'n_target', 'effective_half_width',
    'sigma_masking', 'sigma_accuracy', 'sizing_metric', 'sampling_mode'
]

with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for p, (level, r) in zip(P_VALUES, results.items()):
        writer.writerow({
            'network': NETWORK_NAME,
            'dataset': DATASET,
            'p_value': p,
            'injection_level': level,
            'golden_accuracy': r['golden_accuracy'],
            'mean_masking_ratio': r['mean_masking_ratio'],
            'std_masking_ratio': r['std_masking_ratio'],
            'mean_critical_ratio': r['mean_critical_ratio'],
            'std_critical_ratio': r['std_critical_ratio'],
            'mean_faulty_accuracy': r['mean_faulty_accuracy'],
            'n_trials': r['n_trials'],
            'n_target': r['n_target'],
            'effective_half_width': r['effective_half_width'],
            'sigma_masking': r['sigma_masking'],
            'sigma_accuracy': r['sigma_accuracy'],
            'sizing_metric': r['sizing_metric'],
            'sampling_mode': r['sampling_mode'],
        })

# ----------------------------------------------------------------
# Save human-readable summary
# ----------------------------------------------------------------
output_txt = os.path.join(output_dir, f'{NETWORK_NAME}_{DATASET}_ber.txt')

with open(output_txt, 'w') as f:
    f.write(f'BER Campaign Summary\n')
    f.write(f'====================\n')
    f.write(f'Network : {NETWORK_NAME}\n')
    f.write(f'Dataset : {DATASET}\n')
    f.write(f'Mode    : {list(results.values())[0]["sampling_mode"]}\n')
    f.write(f'Golden accuracy: {list(results.values())[0]["golden_accuracy"]:.4f}\n')
    f.write(f'\n')
    f.write(f'{"p_value":<10} {"k_bits":<8} {"M":<8} {"std_M":<8} {"crit":<8} {"acc":<8} {"n_trials":<9} {"half_w":<8}\n')
    f.write(f'{"-"*70}\n')
    for p, (level, r) in zip(P_VALUES, results.items()):
        f.write(f'{p:<10.0e} {level:<8} {r["mean_masking_ratio"]:<8.4f} '
                f'{r["std_masking_ratio"]:<8.4f} {r["mean_critical_ratio"]:<8.4f} '
                f'{r["mean_faulty_accuracy"]:<8.4f} {r["n_trials"]:<9} '
                f'{r["effective_half_width"]:<8.4f}\n')

print(f'Results saved to: {output_csv}')
print(f'Summary saved to: {output_txt}')
