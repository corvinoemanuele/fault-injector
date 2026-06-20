#!/bin/bash

conda activate nn_rel
cd /home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI/training

echo "=== Inizio training: $(date) ==="

python3 train.py --network SHUFFLENETV2_X1_0 --lr 0.0001 --gamma 0.8 --step_lr 10
python3 train.py --network SHUFFLENETV2_X1_0 --lr 0.0005 --gamma 0.8 --step_lr 10
python3 train.py --network SHUFFLENETV2_X1_0 --lr 0.001 --gamma 0.8 --step_lr 10


echo "=== Fine training: $(date) ==="