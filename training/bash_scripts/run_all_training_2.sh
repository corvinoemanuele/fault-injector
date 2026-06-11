#!/bin/bash

conda activate nn_rel
cd /home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI/training

echo "=== Inizio training: $(date) ==="

python3 train.py --lr 1 --network RESNET56
python3 train.py --lr 1.1 --network RESNET56
python3 train.py --lr 1.2 --network RESNET56


echo "=== Fine training: $(date) ==="