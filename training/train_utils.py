from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv
import json

import os
import sys

ROOT = '/home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI'
sys.path.insert(0, f'{ROOT}/models')

import nicolo_net

def train(model, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    total_loss = 0
    correct = 0
    parameter_updates = 0

    with tqdm(train_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            parameter_updates += 1
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            tepoch.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    return avg_loss, accuracy, parameter_updates


def test(model, device, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return avg_loss, accuracy

class EarlyStopper:
    ''' 
    the training stop itself if there are more than n consecutive epochs without any improvment in accuracy 
    
    '''
    def __init__(self, patience=201, min_delta=0.00001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta): 
            self.counter += 1
            if self.counter > self.patience:
                return True
        return False

class BatchSizeScheduler:
    def __init__(self, batch_size, min_batch_size, max_batch_size, step_size, gamma):
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.step_size = step_size
        self.gamma = gamma

    def get_batch_size(self, epoch):
        new_batch_size = self.batch_size * (self.gamma ** (epoch // self.step_size))
        new_batch_size = max(new_batch_size, self.min_batch_size)
        new_batch_size = min(new_batch_size, self.max_batch_size)
        return int(new_batch_size)

class ExperimentLogger:
    """
    Handles all logins for a single train run:
    - per epoch CSV
    - TensorBoard
    - global experiments log
    - manifest JSON
    """
    CSV_HEADERS = [
        'epoch', 'train_loss', 'train_acc',
        'val_loss', 'val_acc', 'val_split',
        'test_loss', 'test_acc',
        'lr', 'batch_size', 'parameter_updates'
    ]

    EXPERIMENTS_LOG_HEADERS = [
        "run_id", "date", "network", "dataset", "optimizer",
        "lr", "batch_size", "gamma", "weight_decay",
        "epochs", "epochs_completed",
        "final_train_acc", "final_test_acc",
        "sparsity", "scheduler"
    ]

    def __init__(self, csv_path: str, tensorboard_writer, experiments_log: str):
        """
        Args:
            csv_path: path to per-epoch CSV file.
            tensorboard_writer: SummaryWriter instance.
            experiments_log: path to global experiments log CSV.
        """
        self.csv_path = csv_path
        self.writer = tensorboard_writer
        self.experiments_log = experiments_log
        self._init_csv()
    
    def _init_csv(self):
        """Initialize per-epoch CSV with fixed headers."""
        with open(self.csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(self.CSV_HEADERS)

    def log_epoch(self, epoch,
                  train_loss, train_acc,
                  test_loss, test_acc,
                  lr, batch_size, updates,
                  val_loss=None, val_acc=None, val_split=None):
        """Log one epoch to CSV and TensorBoard."""

        # CSV
        row = [
            epoch, train_loss, train_acc,
            val_loss, val_acc, val_split,
            test_loss, test_acc,
            lr, batch_size, updates
        ]
        with open(self.csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        # TensorBoard
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Accuracy/train", train_acc, epoch)
        self.writer.add_scalar("Loss/test", test_loss, epoch)
        self.writer.add_scalar("Accuracy/test", test_acc, epoch)
        if val_loss is not None:
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
        self.writer.flush()

    def log_final(self, run_id, date, network_name, dataset,
                  optimizer_name, lr, batch_size, gamma, weight_decay,
                  epochs, epoch, train_acc, test_acc,
                  sparsity, scheduler):
        """Append run summary to global experiments log."""
        log_exists = os.path.exists(self.experiments_log)
        with open(self.experiments_log, 'a', newline='') as f:
            writer = csv.writer(f)
            if not log_exists:
                writer.writerow(self.EXPERIMENTS_LOG_HEADERS)
            writer.writerow([
                run_id, date, network_name, dataset, optimizer_name,
                lr, batch_size, gamma, weight_decay,
                epochs, epoch,
                train_acc, test_acc,
                sparsity, scheduler
            ])
    
    def save_manifest(self, manifest_path: str, manifest: dict):
        """Save run manifest to JSON."""
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()



def get_network_untrained(network_name, device, dataset_name):
    
    if dataset_name == 'CIFAR100':
        if 'RESNET56' in network_name:
            network = nicolo_net.cifar_resnet56(num_classes=100)
        elif 'VGG13' in network_name:
            network = nicolo_net.cifar_vgg13_bn(num_classes=100)
        elif 'MOBILENETV2_X1_0' in network_name:
            network = nicolo_net.cifar_mobilenetv2_x1_0(num_classes=100)
        elif 'MOBILENETV2_X1_4' in network_name:
            network = nicolo_net.cifar_mobilenetv2_x1_4(num_classes=100)
        elif 'SHUFFLENETV2' in network_name:
            network = nicolo_net.cifar_shufflenetv2_x2_0(num_classes=100)
        else:
            raise ValueError(f'Rete {network_name} non supportata per il training')
    else:
        raise ValueError(f'Dataset {dataset_name} non supportato per il training')

    network = network.to(device)
    return network

def measure_sparsity(model, dataloader, device, threshold=0.0, quantile=0.99):
    """
    Measure sparsity of activations in the second to last layer.

    """
    model.eval()
    model.to(device)

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    feature_extractor.to(device)

    embeddings = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            output = feature_extractor(images)
            output = output.flatten(start_dim=1)
            embeddings.append(output.cpu().numpy())

    embeddings = np.concatenate(embeddings)

    percentile_per_channel = np.quantile(np.abs(embeddings), quantile, axis=0)

    inactive = np.sum(percentile_per_channel <= threshold) 
    Nf = embeddings.shape[1]
    sparsity = inactive / Nf

    feature_extractor.to('cpu')
    del feature_extractor, embeddings, output
    import gc
    gc.collect()

    return sparsity

def setup_loaders(network_name, batch_size, dataset, dataset_root):
    """Load train and test loaders."""
    from utils import get_loader
    train_loader, test_loader = get_loader(
        network_name=network_name,
        batch_size=batch_size,
        dataset_name=dataset,
        root=dataset_root
    )
    return train_loader, test_loader

def setup_validation(train_loader, val_split, batch_size):
    """Split train loader into train and validation subsets."""
    dataset_obj = train_loader.dataset
    val_size    = int(len(dataset_obj) * val_split)
    train_size  = len(dataset_obj) - val_size
    train_set, val_set = torch.utils.data.random_split(
        dataset_obj, [train_size, val_size]
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )
    print(f"Validation split: {train_size} train / {val_size} val ({val_split*100:.0f}%)")
    return train_loader, val_loader

def setup_optimizer(optimizer_name, model, lr, weight_decay):
    """Create optimizer."""
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr,
                         weight_decay=weight_decay, 
                         momentum=0.9, nesterov=True)
    else:
        raise ValueError(f'Optimizer {optimizer_name} not supported.')

def setup_scheduler(scheduler_name, optimizer, step_lr, gamma, epochs, lr):
    """Create LR scheduler."""
    if scheduler_name == 'stepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_lr, gamma=gamma)
    elif scheduler_name == 'cosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    elif scheduler_name == 'cosineAnnealingLR_warmup':
        warmup_epochs = 5
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,  # parte da LR * 0.001 = 0.0001
            end_factor=1.0,      # arriva a LR pieno
            total_iters=warmup_epochs
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=0
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs]
        )
    elif scheduler_name == 'cosineAnnealingLR_constant':
        cosine_epochs = epochs - 50
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=0.001
        )
        constant = optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=0.001 / lr,
            total_iters=50
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[cosine, constant],
            milestones=[cosine_epochs]
        )
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported. Use stepLR, cosineAnnealingLR or cosineAnnealingLR_warmup.")

def get_next_run_number(base_dir: str):
    """
    Progressive counter of experiments in the network's folder. It scans the folder (e.g. trained_models/CIFAR100/RESNET56/ and gives back max + 1. It ignores folders which don't start with a number (e.g. _archive). 
    """

    if not os.path.exists(base_dir):
        return 1

    existing = [d for d in os.listdir(base_dir) 
                if  os.path.isdir(os.path.join(base_dir, d))]

    numbers = []
    for d in existing:
        try:
            n = int(d.split('_')[0])
            numbers.append(n)
        except (ValueError, IndexError):
            continue

    return max(numbers, default=0) + 1

def build_run_path(root, dataset, network_name, optimizer_name,
                   lr_str, batch_size, epochs, results_dir):
    """
    Builds and creates all directories for a single training run.
    Strucure : trained_models/{dataset}/{network_name}/{run_id}/

    Args:
        root: project root path
        dataset: dataset name (e.g. 'CIFAR100')
        network_name: network name (e.g. 'RESNET56')
        optimizer_name: optimizer name (e.g. 'adam')
        lr_str: learning rate as string without dot (e.g. 0.01 -> '001')
        batch_size: training batch size
        epochs: number of epochs
        results_dir: path to results folder for per-epoch CSV files

    Returns:
        dict with keys:
            run_id: full run identifier (e.g. '001_ADAM_LR_001_BS_128_EP_200')
            run_num: progressive run number
            run_folder: run folder inside trained_models/
            run_tensorboard_dir: run folder inside tensorboard/ for TensorBoard
            model_path: full path to the .pt file
            manifest_path: full path to manifest.json
            csv_path: full path to per-epoch CSV in results/
    """
    run_name = f'{optimizer_name.upper()}_LR_{lr_str}_BS_{batch_size}_EP_{epochs}'

    network_models_dir = f'{root}/training/trained_models/{dataset}/{network_name}/'
    network_tensorboard_dir = f'{root}/training/tensorboard/{dataset}/{network_name}'

    os.makedirs(network_models_dir, exist_ok=True)
    os.makedirs(network_tensorboard_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Run identifier
    run_num = get_next_run_number(network_models_dir)
    run_id = f'{run_num:03d}_{run_name}'

    # Run-level directories
    run_folder = f'{network_models_dir}/{run_id}'
    run_tensorboard_dir = f'{network_tensorboard_dir}/{run_id}'

    os.makedirs(run_folder)
    os.makedirs(run_tensorboard_dir, exist_ok=True)

    return {
        'run_id': run_id,
        'run_num': run_num,
        'run_folder': run_folder,
        'run_tensorboard_dir': run_tensorboard_dir,
        'model_path': f'{run_folder}/{network_name}_{optimizer_name.upper()}_EP_{epochs}_LR_{lr_str}_{dataset}.pt',
        'manifest_path': f'{run_folder}/manifest.json',
        'csv_path': f'{results_dir}/{network_name}_{run_id}.csv'
    }