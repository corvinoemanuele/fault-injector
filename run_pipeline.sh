#!/bin/bash

cd /home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI
conda activate nn_rel

echo "=== Step 1: Generating fault list ==="
python3 fault_list_gen_v2.py
if [ $? -ne 0 ]; then
    echo "ERROR: fault_list_gen_v2.py failed. Stopping."
    exit 1
fi

echo "=== Step 2: Running fault injection ==="
python3 run_fi.py
if [ $? -ne 0 ]; then
    echo "ERROR: run_fi.py failed. Stopping."
    exit 1
fi

echo "=== Step 3: Analyzing results ==="
python3 data_analyzer_no_writing_cython_serial_incremental.py
if [ $? -ne 0 ]; then
    echo "ERROR: data_analyzer failed. Stopping."
    exit 1
fi

echo "=== Pipeline completed successfully ==="