#!/bin/bash

conda activate nn_rel
cd /home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI/training

echo "=== Inizio training: $(date) ==="

python3 train.py --lr 0.002
python3 train.py --lr 0.0005
python3 train.py --lr 0.001
python3 train.py --lr 0.0015



echo "=== Fine training: $(date) ==="

