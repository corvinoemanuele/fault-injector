import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import csv
import json
import time
import os
import sys

ROOT = '/home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from utils import get_loader
from train_config import (
    NETWORK_NAME, DATASET, ROOT, LR, WEIGHT_DECAY, EPOCHS, BATCH_SIZE,
    OPTIMIZER, SCHEDULER, STEP_LR, GAMMA, DATASET_ROOT, RESULTS_DIR,
    USE_VALIDATION, VAL_SPLIT, SEED, DEVICE,
    USE_BATCH_SCHEDULER, MAX_BATCH_SIZE, MIN_BATCH_SIZE,
    STEP_BATCH_SIZE, GAMMA_BATCH_SIZE, EARLY_STOPPING_MIN_DELTA, EARLY_STOPPING_PATIENCE
)
from train_utils import (
    train, test,
    EarlyStopper, BatchSizeScheduler, ExperimentLogger,
    get_network_untrained, measure_sparsity,
    setup_loaders, setup_validation, setup_optimizer, setup_scheduler,
    build_run_path
)
import argparse

DATE_TAG = time.strftime("%Y-%m-%d")
EXPERIMENTS_LOG = f'{ROOT}/training/results/experiments_log.csv'

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for experiments')
    
    parser.add_argument('--network', type=str, default=NETWORK_NAME)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--gamma', type=float, default=GAMMA)
    parser.add_argument('--step_lr', type=int, default=STEP_LR)
    parser.add_argument('--scheduler', type=str, default=SCHEDULER)
    parser.add_argument('--dry_run', action='store_true', default=False, help='If True data and folders are not saved')
    
    return parser.parse_args()

def main():
    #Setting parser
    args = parse_args()
    network_name = args.network
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    optimizer_name = args.optimizer
    weight_decay = args.weight_decay
    gamma = args.gamma
    step_lr = args.step_lr
    scheduler_name = args.scheduler
    dry_run = args.dry_run

    if dry_run:
        print("[DRY_RUN] File will not be saved.")
    
    #Training settings for reproducibility
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(DEVICE)

    lr_str = str(lr).replace('.', '')

    #Paths & logger
    if not dry_run:
        path = build_run_path(
            root=ROOT,
            dataset=DATASET,
            network_name=network_name,
            optimizer_name=optimizer_name,
            lr_str=lr_str,
            batch_size=batch_size,
            epochs=epochs,
            results_dir=RESULTS_DIR
        )
        writer = SummaryWriter(path['run_tensorboard_dir'])
        logger = ExperimentLogger(
            csv_path=path['csv_path'],
            tensorboard_writer=writer,
            experiments_log=EXPERIMENTS_LOG
        )
        print(f"Run: {path['run_id']}")
    else:
        path = {'run_id': 'DRY_RUN'}
        logger = None
        writer = None
        print(f"[DRY RUN] LR={lr} | BS={batch_size} | EP={epochs}")

    #Model and dataset loading
    model = get_network_untrained(network_name, device, DATASET)

    train_loader, test_loader = setup_loaders(
        network_name, batch_size, DATASET, DATASET_ROOT
    )
    data, target = next(iter(train_loader))
    print(f"Network: {network_name}")
    print(f"Data shape: {data.shape} | Target sample: {target[:5].tolist()}")
    
    val_loader = None
    if USE_VALIDATION:
        train_loader, val_loader = setup_validation(
            train_loader, VAL_SPLIT, batch_size
        )

    #Components
    loss_fn = nn.CrossEntropyLoss()
    optimizer = setup_optimizer(optimizer_name, model, lr, weight_decay)
    scheduler = setup_scheduler(scheduler_name, optimizer, step_lr, gamma, epochs, lr)
    early_stopper = EarlyStopper(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA
    )

    if USE_BATCH_SCHEDULER:
        batch_scheduler = BatchSizeScheduler(
            batch_size=batch_size,
            min_batch_size=MIN_BATCH_SIZE,
            max_batch_size=MAX_BATCH_SIZE,
            step_size=STEP_BATCH_SIZE,
            gamma=GAMMA_BATCH_SIZE
        )

    #Training loop
    updates_per_epoch = []

    for epoch in range(1, epochs + 1):
        
        train_loss, train_acc, updates = train(
            model, device, train_loader, optimizer, loss_fn, epoch
        )
        updates_per_epoch.append(updates)

        val_loss, val_acc = None, None
        if USE_VALIDATION:
            val_loss, val_acc = test(model, device, val_loader, loss_fn)

        #Test metrics for monitoring only
        test_loss, test_acc = test(model, device, test_loader, loss_fn)

        if scheduler is not None:
            scheduler.step()

        current_batch_size = batch_size
        if USE_BATCH_SCHEDULER:
            current_batch_size = batch_scheduler.get_batch_size(epoch)
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset, 
                batch_size=current_batch_size, 
                shuffle=True
            )
        
        # Print metrics for each epoch
        log_line = (f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if USE_VALIDATION:
            log_line += f" | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"

        print(log_line)

        # Logging metrics into csv using ExperimentLogger 
        if not dry_run:
            logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss, train_acc=train_acc,
                test_loss=test_loss,   test_acc=test_acc,
                lr=optimizer.param_groups[0]['lr'],
                batch_size=current_batch_size,
                updates=updates,
                val_loss=val_loss, val_acc=val_acc,
                val_split=VAL_SPLIT if USE_VALIDATION else None
            )
        
        if USE_VALIDATION and early_stopper.early_stop(val_loss):
            print(f'Early stopping at epoch {epoch}')
            break

    print(f'Updates for epoch: {updates_per_epoch}\n')

    print("Measuring sparsity...")
    final_sparsity = measure_sparsity(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=0.0,
        quantile=0.99
    )
    print(f"Sparsity: {final_sparsity * 100:.2f}%")

    if dry_run:
        print("[DRY RUN] Training completed. No files saved.")
        return
        
    # Save model
    torch.save(model.state_dict(), path['model_path'])
    print(f"Model saved: {path['model_path']}")

    # Save manifest
    manifest = {
        "run_id": path['run_id'],
        "date": DATE_TAG,
        "run_number": path['run_num'],
        "network": network_name,
        "dataset": DATASET,
        "optimizer": optimizer_name,
        "lr": lr,
        "scheduler": scheduler_name,
        "step_lr": step_lr,
        "gamma": gamma,
        "batch_size": batch_size,
        "use_validation": USE_VALIDATION,
        "val_split": VAL_SPLIT if USE_VALIDATION else None,
        "use_batch_scheduler": USE_BATCH_SCHEDULER,
        "max_batch_size": MAX_BATCH_SIZE if USE_BATCH_SCHEDULER else None,
        "min_batch_size": MIN_BATCH_SIZE if USE_BATCH_SCHEDULER else None,
        "step_batch_size": STEP_BATCH_SIZE if USE_BATCH_SCHEDULER else None,
        "gamma_batch_size": GAMMA_BATCH_SIZE if USE_BATCH_SCHEDULER else None,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "epochs_completed": epoch,
        "seed": SEED,
        "final_train_loss": train_loss,
        "final_train_acc": train_acc,
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        "sparsity": final_sparsity,
        "model_path": path['model_path'],
        "csv_path": path['csv_path'],
        "notes": ""
    }
    logger.save_manifest(path['manifest_path'], manifest)
    print(f"Manifest saved: {path['manifest_path']}")

    # Global log
    logger.log_final(
        run_id=path['run_id'], date=DATE_TAG,
        network_name=network_name, dataset=DATASET,
        optimizer_name=optimizer_name,
        lr=lr, batch_size=batch_size,
        gamma=gamma, weight_decay=weight_decay,
        scheduler=scheduler_name,
        epochs=epochs, epoch=epoch,
        train_acc=train_acc, test_acc=test_acc,
        sparsity=final_sparsity
    )

    logger.close()

if __name__ == '__main__':
    main()