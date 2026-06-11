import torch
import os
import sys
import csv
import copy
import random
import numpy as np

sys.path.insert(0, '/home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI')
os.chdir('/home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI')

from config import (
    ROOT, NETWORK_NAME, DATASET, DATASET_ROOT, RESULTS_ROOT, FAULT_MODEL , SEED, MAX_FAULTS_TO_INJECT, BATCH_SIZE, PRINT
)
from utils import get_network, get_loader

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Carica la rete — il NETWORK_NAME in config contiene già eventuali suffissi _PRUNED, _TMR
    network = get_network(
        network_name=NETWORK_NAME,
        device=device,
        dataset_name=DATASET,
        root=ROOT
    )
    network.eval()

    # Calcola parametri totali e per layer
    total_sum_params = sum(
        p.numel() for p in network.parameters() if len(p.size()) >= 2
    )
    layer_params_list = [
        (name, p.numel())
        for name, p in network.named_parameters()
        if len(p.size()) >= 2
    ]
    layer_dimensions_list = [
        (name, np.array(p.size()))
        for name, p in network.named_parameters()
        if len(p.size()) >= 2
    ]

    if PRINT:
        print(f'Parametri totali: {total_sum_params}')
        for name, params in layer_params_list:
            print(f'Layer: {name}, Params: {params}')

    # Calcolo numero fault da iniettare con formula statistica
    p_val = 0.5
    e = 0.01
    t = 2.58
    N = total_sum_params * 32 * 2
    fault_to_inject = round(N / (1 + e**2 * (N - 1) / (t**2 * p_val * (1 - p_val))))
    print(f'Fault totali possibili: {N}')
    print(f'Fault da iniettare: {fault_to_inject}')

    # Distribuzione fault per layer proporzionalmente ai parametri
    faults_to_inject_list = [
        (layer_name, round((total_params * fault_to_inject) / total_sum_params))
        for layer_name, total_params in layer_params_list
    ]

    if PRINT:
        print('\nFault per layer:')
        for layer_name, faults in faults_to_inject_list:
            print(f'Layer: {layer_name}, Faults: {faults}')

    # Generazione fault list
    random.seed(SEED)
    FILE_NAME = f'{SEED}_parameters_fault_list.csv'
    csv_filename = f'output/fault_list/{NETWORK_NAME}/{FILE_NAME}'
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    header = ['Injection', 'Layer', 'TensorIndex', 'Bit']
    counter = 0
    used_indices = set()
    row_number = 0
    max_attempts = 1000

    with open(csv_filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)

        for injection_number, (layer_name, _) in enumerate(layer_params_list):
            layer_dimensions = next(
                dims for name, dims in layer_dimensions_list if name == layer_name
            )
            for _ in range(faults_to_inject_list[injection_number][1]):
                attempts = 0
                while attempts < max_attempts:
                    if len(layer_dimensions) == 4:
                        idx = tuple(random.randint(0, d - 1) for d in layer_dimensions)
                        tensor_index = f'({idx[0]}, {idx[1]}, {idx[2]}, {idx[3]})'
                    elif len(layer_dimensions) == 3:
                        idx = tuple(random.randint(0, d - 1) for d in layer_dimensions)
                        tensor_index = f'({idx[0]}, {idx[1]}, {idx[2]})'
                    elif len(layer_dimensions) == 2:
                        idx = tuple(random.randint(0, d - 1) for d in layer_dimensions)
                        tensor_index = f'({idx[0]}, {idx[1]})'

                    bit_flip = random.randint(0, 31)
                    layer_name_clean = layer_name.replace('.weight', '')
                    index_key = (layer_name_clean, tensor_index, bit_flip)

                    if index_key not in used_indices:
                        break
                    attempts += 1
                    counter += 1

                if attempts == max_attempts:
                    print(f'Impossibile trovare combinazione unica per {layer_name}')
                    break

                used_indices.add(index_key)
                csv_writer.writerow([row_number, layer_name_clean, tensor_index, bit_flip])
                row_number += 1

    print(f'Fault list salvata in: {csv_filename}')
    print(f'Fault unici generati: {len(used_indices)}')
    print(f'Tentativi duplicati: {counter}')

if __name__ == '__main__':
    main()