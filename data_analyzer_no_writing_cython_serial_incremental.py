import gc
import torch
from tqdm import tqdm
import numpy as np
from torchvision.transforms.v2 import ToTensor,Resize,Compose,ColorJitter,RandomRotation,AugMix,RandomCrop,GaussianBlur,RandomEqualize,RandomHorizontalFlip,RandomVerticalFlip
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST,CIFAR100, GTSRB
import shutil
import os
import cython
import sys
sys.path.append('..')


from utils import get_loader as get_loader_vit


def analyze_FI_output(loader, batch_size, clean_output_path, faulty_output_path,output_path,write_images=False):
    
    n_batches = len(loader)
    dataset_size = len(loader.dataset)

    batch_size = batch_size
    pbar = tqdm(loader,
                colour='green',
                desc=f'Saving test labels',
                ncols=shutil.get_terminal_size().columns)
    
    # Initialize an empty list to store batch information
    batch_info_array = np.zeros((n_batches, int(batch_size)), dtype=np.int_)
    for batch_id, batch in enumerate(pbar):
        _, label = batch
        batch_info_array[batch_id, :label.shape[0]] = label.numpy()
  
    gc.collect()
    #del batch_info
    gc.collect()
    # Load clean tensor
    loaded_clean_output_ = np.load(clean_output_path, allow_pickle=True)
    last_batch_size = loaded_clean_output_[-1].shape[0]
    
    loaded_clean_output_[-1] = np.append(loaded_clean_output_[-1], np.zeros((loaded_clean_output_[0].shape[0] - loaded_clean_output_[-1].shape[0], loaded_clean_output_[-1].shape[1]),dtype=np.float32), axis=0)
    
    
    loaded_clean_output = np.concatenate(loaded_clean_output_, axis=0)
    loaded_clean_output = loaded_clean_output.reshape(n_batches, int(batch_size), loaded_clean_output_[0].shape[-1])
    print(f'{loaded_clean_output.shape=}')
    
    del loaded_clean_output_
    gc.collect()
    print(f'number of batch: {n_batches}')

    
    print(f'Starting output definition: {datetime.now()}')
    

    clean_output_match_counter = 0
    faulty_output_match_counter = 0
    clean_output_match_counter_sdc5 = 0
    faulty_output_match_counter_sdc5 = 0
    masked = 0
    best5_c = 0
    critical = 0
    not_critical = 0
    
    #inside faults
    
    fname = os.path.join(output_path,f'fault_classification.csv')
    
    print(f'output_path: {fname}')
    
    if os.path.exists(fname):
        print(f'WARNING! {fname} already exists. Removing it?')
        choice = input('y/n: ')
        if choice == 'y':
            os.remove(fname)
        else:
            print('Exiting...')
            exit()
            
    start_from = 0
    for i in range(n_batches):
        #Load the faulty_output
        
        print(f'Starting batch {i}: {datetime.now()}')
        
        faulty_output_filename = os.path.join(faulty_output_path, f'batch_{i}.npy')
        loaded_faulty_output = np.load(faulty_output_filename)

        faulty_tensor_data = loaded_faulty_output
        n_faults = faulty_tensor_data.shape[0]
        n_classes = faulty_tensor_data.shape[-1]
        
        # iterate over the faults in the batch
        for z in tqdm(range(n_faults), desc="fault progress"):
            dim_batch =  batch_info_array[i].shape[0] if i < n_batches-1 else last_batch_size
                
            res = process_a_fault_writing(z, i, n_classes, dim_batch, start_from, loaded_clean_output[i], faulty_tensor_data, batch_info_array[i],fname.encode('ascii'),write_images)
            clean_output_match_counter_local, faulty_output_match_counter_local, clean_output_match_counter_sdc5_local, faulty_output_match_counter_sdc5_local, best5_c_local, masked_local, critical_local, not_critical_local = res
            # Add local counters to global counters
            clean_output_match_counter += clean_output_match_counter_local
            faulty_output_match_counter += faulty_output_match_counter_local
            clean_output_match_counter_sdc5 += clean_output_match_counter_sdc5_local
            faulty_output_match_counter_sdc5 += faulty_output_match_counter_sdc5_local
            best5_c += best5_c_local
            masked += masked_local
            critical += critical_local
            not_critical += not_critical_local
            
        start_from += batch_info_array[i].shape[0] if i < n_batches-1 else last_batch_size
            
            
    print(f'Ending output definition: {datetime.now()}')                                                   
    # print the results
    print(f'total outputs: {masked + not_critical + critical}')
    print('masked:', masked)
    print(f'% masked faults: {100*masked/(masked + not_critical + critical)} %')
    print('not critical faults:', not_critical)
    print(f'% not critical: {100*not_critical/(masked + not_critical + critical)} %')
    print('critical faults:', critical)
    print(f'% critical: {100*critical/(masked + not_critical + critical)} %')
    print('criticial faults in 5:', best5_c)
    print(f'% critical faults in 5: {(best5_c / (masked + not_critical + critical)) * 100} %')

    print(f'\nTOP-1 clean accuracy: {100*clean_output_match_counter / (dataset_size*(n_faults+1))}')
    print(f'TOP-1 faulty accuracy: {100*faulty_output_match_counter / (dataset_size*(n_faults+1))}')
    
    print(f'\nTOP-5 clean accuracy: {100*clean_output_match_counter_sdc5 / (dataset_size*(n_faults+1))}')
    print(f'TOP-5 faulty accuracy: {100*faulty_output_match_counter_sdc5 / (dataset_size*(n_faults+1))}')

    with open(output_file, 'w') as out_file:
        with contextlib.redirect_stdout(out_file):
            print(f'Batch size: {batch_size}')
            print(f'total outputs: {masked + not_critical + critical}')
            print('masked:', masked)
            print(f'% masked faults: {100*masked/(masked + not_critical + critical)} %')
            print('not critical faults:', not_critical)
            print(f'% not critical: {100*not_critical/(masked + not_critical + critical)} %')
            print('critical faults:', critical)
            print(f'% critical: {100*critical/(masked + not_critical + critical)} %')
            print('criticial faults in 5:', best5_c)
            print(f'% critical faults in 5: {(best5_c / (masked + not_critical + critical)) * 100} %')

            print(f'\nTOP-1 clean accuracy: {100*clean_output_match_counter / (dataset_size*(n_faults+1))}')
            print(f'TOP-1 faulty accuracy: {100*faulty_output_match_counter / (dataset_size*(n_faults+1))}')
            
            print(f'\nTOP-5 clean accuracy: {100*clean_output_match_counter_sdc5 / (dataset_size*(n_faults+1))}')
            print(f'TOP-5 faulty accuracy: {100*faulty_output_match_counter_sdc5 / (dataset_size*(n_faults+1))}')

import sys
import os

# Add build directory to path for process_batch_serial
build_dir = os.path.join(os.path.dirname(__file__), 'build')
# Try to find any available Python version build
if os.path.exists(build_dir):
    for lib_dir in os.listdir(build_dir):
        if lib_dir.startswith('lib.linux-x86_64'):
            full_path = os.path.join(build_dir, lib_dir)
            if os.path.exists(full_path) and full_path not in sys.path:
                sys.path.insert(0, full_path)
                break

#import cython_process_batch
#from cython_process_batch import process_batch_serial
import process_batch_serial
from process_batch_serial import process_a_batch, process_a_fault, process_a_fault_writing
    
import contextlib
from datetime import datetime

from config import (
    NETWORK_NAME, DATASET, BATCH_SIZE, SEED, DATASET_ROOT, RESULTS_ROOT,
    SEED_IMAGENET, FAULT_MODEL, DELETE_FAULTY_OUTPUT
)

train_split = 1

# Opzioni raramente modificate — lasciate qui invece che in config
REDUCED = False
EMBEDDING = True
WRITE_IMAGES_RESPONSE = True

# Path input/output
clean_output_path = f'./output/clean_output/{DATASET}/{NETWORK_NAME}/batch_{BATCH_SIZE}/clean_output.npy'
faulty_output_path = f'./output/faulty_output/{DATASET}/{NETWORK_NAME}/batch_{BATCH_SIZE}/{FAULT_MODEL}/{SEED}'
out_dir = os.path.join(RESULTS_ROOT, f'{NETWORK_NAME}_fault_classification')

if EMBEDDING:
    out_dir = os.path.join(out_dir, 'EMBEDDING', f'FAULT_LIST_RS_{SEED}')

output_file = os.path.join(
    out_dir,
    f'{NETWORK_NAME}_fault_classification.out' if not REDUCED else 'reduced_fault_classification.out'
)

# Carica il loader
if DATASET == 'ImageNet':
    train_loader, test_loader = get_loader_vit(
        network_name=NETWORK_NAME,
        batch_size=BATCH_SIZE,
        dataset_name=DATASET,
        root=DATASET_ROOT,
        train_split=SEED_IMAGENET
    )
else:
    train_loader, test_loader = get_loader_vit(
        network_name=NETWORK_NAME,
        batch_size=BATCH_SIZE,
        dataset_name=DATASET,
        root=DATASET_ROOT
    )

os.makedirs(out_dir, exist_ok=True)

analyze_FI_output(test_loader, BATCH_SIZE, clean_output_path, faulty_output_path, out_dir, WRITE_IMAGES_RESPONSE)

if DELETE_FAULTY_OUTPUT and os.path.isdir(faulty_output_path):
    shutil.rmtree(faulty_output_path)
    print(f'Deleted {faulty_output_path}')


