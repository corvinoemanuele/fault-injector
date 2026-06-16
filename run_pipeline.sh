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

echo "=== Step 4: Cleaning up faulty_output folder ==="
# Extract the dataset and network name from config to construct the path
DATASET=$(grep "^DATASET = " config.py | sed "s/DATASET = '\(.*\)'/\1/")
NETWORK_NAME=$(grep "^NETWORK_NAME = " config.py | sed "s/NETWORK_NAME = '\(.*\)'/\1/")
BATCH_SIZE=$(grep "^BATCH_SIZE = " config.py | sed "s/BATCH_SIZE = \(.*\)/\1/")
FAULT_MODEL=$(grep "^FAULT_MODEL = " config.py | sed "s/FAULT_MODEL = '\(.*\)'/\1/")
SEED=$(grep "^SEED = " config.py | sed "s/SEED = \(.*\)/\1/")

FAULTY_OUTPUT_FOLDER="./output/faulty_output/${DATASET}/${NETWORK_NAME}/batch_${BATCH_SIZE}/${FAULT_MODEL}/${SEED}"

if [ -d "$FAULTY_OUTPUT_FOLDER" ]; then
    echo "Deleting $FAULTY_OUTPUT_FOLDER"
    rm -rf "$FAULTY_OUTPUT_FOLDER"
    echo "Cleanup completed!"
else
    echo "Faulty output folder not found at: $FAULTY_OUTPUT_FOLDER"
fi

echo "=== Pipeline completed successfully ==="