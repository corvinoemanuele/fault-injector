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
    injection_levels=[1], sampling_mode='constant', seed=SEED
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
output_dir = os.path.join(ROOT, 'output', 'ber_results')
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, f'{NETWORK_NAME}_{DATASET}_ber.csv')

fieldnames = [
    'network', 'dataset', 'p_value', 'injection_level',
    'golden_accuracy', 'mean', 'std', 'n_trials', 'n_target',
    'effective_half_width', 'sampling_mode'
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
            'mean': r['mean'],
            'std': r['std'],
            'n_trials': r['n_trials'],
            'n_target': r['n_target'],
            'effective_half_width': r['effective_half_width'],
            'sampling_mode': r['sampling_mode'],
        })

print(f'\nResults saved to: {output_csv}')