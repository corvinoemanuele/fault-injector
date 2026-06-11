
````markdown
# Training

This module trains CNN models on CIFAR100 in preparation for Fault Injection (FI) campaigns. The main goals are:

- producing models with controlled activation sparsity by varying the learning rate
- finding the right hyperparameters for custom architectures before running a full FI campaign
- tracking all experiments in a structured and reproducible way

Trained models are saved in `trained_models/` as staging. Once validated, they are manually copied to `models/CIFAR100/pretrained/` for use in the FI pipeline.

---

## Structure

```
training/
├── train.py                  — main training script
├── train_config.py           — hyperparameters and constants
├── train_utils.py            — training functions and classes
├── run_all_training.sh       — launches multiple runs in sequence
├── list_experiments.ipynb    — displays experiments_log.csv as a table
├── metric_extractor.ipynb    — utility for extracting metrics from results
├── trained_models/           — staging: models saved after each run
│   └── CIFAR100/
│       └── {NETWORK}/
│           └── {run_id}/
│               ├── {model}.pt
│               └── manifest.json
├── tensorboard/              — TensorBoard logs
│   └── CIFAR100/
│       └── {NETWORK}/
│           └── {run_id}/
└── results/                  — per-epoch CSV metrics + global log
    ├── experiments_log.csv
    └── CIFAR100/
        └── {NETWORK}_{run_id}.csv
```

---

## File naming convention

Every run is identified by a `run_id` with the following format:

```
{NNN}_{OPTIMIZER}_LR_{LR_STR}_BS_{BS}_EP_{EP}
```

where `NNN` is a zero-padded progressive number per network, and `LR_STR` is the learning rate without the decimal point (`0.01` → `001`).

Examples:

```
001_ADAM_LR_0001_BS_64_EP_150   — run 1, Adam, LR=0.001, BS=64, 150 epochs
013_SGD_LR_01_BS_64_EP_200      — run 13, SGD, LR=0.1, BS=64, 200 epochs
```

The per-epoch CSV follows the same logic:

```
RESNET56_007_ADAM_LR_0001_BS_32_EP_150.csv
```

Parameters not in the name (seed, weight decay, gamma, scheduler) are stored in `manifest.json` inside the run folder.

---

## How to run

### Single run

Set the base parameters in `train_config.py`, then launch:

```bash
python3 training/train.py
```

To override specific parameters without editing the config:

```bash
python3 training/train.py --lr 0.01 --epochs 200 --batch_size 32
```

Available arguments:

```
--network       network architecture (default: from config)
--lr            learning rate (default: from config)
--epochs        number of epochs (default: from config)
--batch_size    batch size (default: from config)
--optimizer     optimizer name: adam or sgd (default: from config)
--weight_decay  weight decay (default: from config)
--gamma         LR scheduler decay factor (default: from config)
--step_lr       LR scheduler step size in epochs (default: from config)
--dry_run       run without saving any file (useful for sanity checks)
```

The typical workflow is: fix the parameters you want to keep constant in `train_config.py`, and use the CLI arguments to vary one or two parameters at a time across experiments.

### Multiple runs in sequence

Edit `run_all_training.sh` to list the runs you want to launch, then:

```bash
bash training/run_all_training.sh
```

This is useful when you want to sweep over learning rates or batch sizes overnight without manual intervention.

### Dry run

To verify that the setup is correct without writing any file:

```bash
python3 training/train.py --dry_run
```

No model, CSV, manifest, or TensorBoard log will be created.

---

## Monitoring

TensorBoard logs are saved automatically during training. To visualize them:

```bash
tensorboard --logdir training/tensorboard/CIFAR100/{NETWORK}
```

---

## After training

Each run produces:

- a `.pt` file with the model weights in `trained_models/CIFAR100/{NETWORK}/{run_id}/`
- a `manifest.json` with all parameters used
- a per-epoch CSV in `results/CIFAR100/`
- a new row appended to `results/experiments_log.csv`

To promote a model to production for use in the FI pipeline, copy the `.pt` file manually to `models/CIFAR100/pretrained/`. Only copy models whose accuracy and sparsity are satisfactory.

It is good practice to revise the results after each training and use the 'notes' section of the manifest to write important results or details to remember.

---

## Edge cases

**Ctrl+C during training:** the run folder in `trained_models/`, the TensorBoard log, and the per-epoch CSV are left on disk but incomplete. No row is added to `experiments_log.csv`. Delete the three artifacts manually to keep the repo aligned:

```bash
rm -rf training/trained_models/CIFAR100/{NETWORK}/{run_id}/
rm -rf training/tensorboard/CIFAR100/{NETWORK}/{run_id}/
rm     training/results/CIFAR100/{NETWORK}_{run_id}.csv
```

**Progressive run numbering:** the run number is assigned at the start of training by scanning the existing folders in `trained_models/CIFAR100/{NETWORK}/`. If a folder was left by an interrupted run, the next run will skip that number. Always clean up interrupted runs before launching new ones.
````

