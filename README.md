# VITTFI — CNN Fault Injection Framework

This project studies how Convolutional Neural Networks degrade in the presence of hardware faults simulated via Fault Injection (FI). Faults are modeled as stuck-at bit flips injected into the network weights during inference. The goal is to analyze fault resilience across different architectures and datasets, classifying each fault as masked, non-critical, or critical, and measuring the resulting accuracy degradation.

The central research hypothesis is that activation sparsity induced by high learning rates correlates linearly with fault masking, following the model M(s) ≈ M0 + α·s (r=0.91). The FI pipeline is the primary tool to validate this hypothesis.

---

## Structure

```
VITTFI/
├── run_fi.py                          — main FI campaign script
├── fault_list_gen_v2.py               — generates fault lists
├── fault_list_gen.ipynb               — legacy, replaced by fault_list_gen_v2.py
├── FaultInjectionManager.py           — manages fault injection during inference
├── FaultGenerators/                   — fault generation utilities
├── OutputFeatureMapsManager/          — experimental, see section below
├── config.py                          — global configuration
├── utils.py                           — model loading, dataset handling
├── data_analyzer_no_writing_cython_serial_incremental.py  — analyzes FI results
├── models/                            — production models organized by dataset
│   ├── CIFAR10/
│   ├── CIFAR100/
│   ├── MNIST/
│   ├── FMNIST/
│   └── ...
├── data/                              — datasets
├── output/                            — FI campaign outputs
│   ├── fault_list/                    — fault lists per network
│   ├── clean_output/                  — golden inference outputs
│   └── faulty_output/                 — faulty inference outputs
├── results/                           — analysis results
│   ├── standard_fi/                   — baseline experiments from the original project
│   ├── ABP/                           — results from pruned networks
│   └── sparsity_fi/                   — LR/sparsity sweep experiments
│       └── ...                        — one folder per network, consider organizing
│                                         by dataset or architecture as experiments grow
├── log/                               — logs
├── misc/                              — miscellaneous files
├── training/                          — training pipeline (see training/README.md)
└── OPT_FI_EXP/                        — advanced experiments: ABP, TMR, DEAR-CNN
```

---

## Model naming convention

Models in `models/{DATASET}/pretrained/` follow this naming convention:

```
{NETWORK}_{OPTIMIZER}_EP_{EPOCHS}_LR_{LR_STR}_{DATASET}.pt
```

where `LR_STR` is the learning rate without the decimal point (`0.01` → `001`).

Examples:
```
RESNET56_ADAM_EP_150_LR_0001_CIFAR100.pt
MOBILENETV2_X1_4_ADAM_EP_200_LR_001_CIFAR100.pt
```

Suffixes indicate model variants:
```
_PRUNED      — network pruned with Activation-Based Pruning
_TMR         — Triple Modular Redundancy applied to the last layer
_PRUNED_TMR  — both
```

Pruned models are saved as checkpoints with the following format:

```python
{
    'state_dict': network.state_dict(),
    'pruned_indices': non_zero_indices,
    'original_network': network_name,
    'dataset': dataset
}
```

and are loaded via `load_pruned_network()` in `utils.py`.

---

## Fault Injection Pipeline

A complete FI campaign consists of three steps run in sequence. A convenience script `run_pipeline.sh` launches all three automatically with error checking.

### Step 1 — Generate the fault list

```bash
python3 fault_list_gen_v2.py
```

The script loads the target model via `get_network()` in `utils.py`, extracts the injectable layers, and generates a fault list defining which weights to perturb and how. The fault list is saved to:

```
output/fault_list/{network_name}/
```

### Step 2 — Run the FI campaign

```bash
python3 run_fi.py
```

Before launching, edit the parameters directly inside `run_fi.py`:
- target network and dataset
- path to the fault list
- path to the clean output
- number of faults to inject
- batch size and device

The script runs golden inference first, stores the clean output, then injects each fault and records the network output. Results are saved to `output/faulty_output/`.

### Step 3 — Analyze results

```bash
python3 data_analyzer_no_writing_cython_serial_incremental.py
```

The analyzer compares golden and faulty outputs, classifies each fault as masked, non-critical, or critical, and writes a summary CSV to `results/sparsity_fi/`.

### Running the full pipeline

To run all three steps in sequence with error checking:

```bash
bash run_pipeline.sh
```

The script stops automatically if any step fails, avoiding partial or inconsistent results in the output folders.

### Adding a new model

1. Place the model definition and weights in `models/{DATASET}/pretrained/`
2. Register the model in `utils.py` inside `get_network()` following the existing pattern
3. Generate a new fault list with `fault_list_gen_v2.py`
4. Run the pipeline

---

## Output Feature Maps (experimental)

`OutputFeatureMapsManager/` captures and stores intermediate feature maps during inference. This component was developed as part of an earlier research direction investigating OFM-level fault metrics, but is not currently used in the main pipeline and did not produce conclusive results.

It is kept for reference. If needed, feature maps are saved as compressed NumPy arrays with the following structure:

```
output/clean_feature_maps/{network}/batch_{size}/batch_{id}_layer_{name}.npz
output/faulty_feature_maps/{network}/batch_{size}/{fault_model}/fault_{id}_batch_{id}_layer_{name}.npz
```

Shape: `B × K × H × W` (batch × channels × height × width). Load with `np.load(file)['arr_0']`.

Network outputs are saved as `.npy` arrays:
- clean: shape `N × B × C` (batches × batch size × classes)
- faulty: shape `F × B × C` (faults × batch size × classes)

Load with `np.load(file, allow_pickle=True)`.

---

## Setup

The project runs on a remote server accessed via SSH. The conda environment `nn_rel` contains all dependencies.

```bash
conda activate nn_rel
cd /home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI
```

The data analyzer uses a Cython-compiled component for performance. If the `.so` file is missing or needs to be rebuilt:

```bash
python3 setup_serial.py build_ext --inplace
```

Required: Python 3.9+, PyTorch, CUDA 11.7, NumPy, Pandas, tqdm.