import gc
import torch
from tqdm import tqdm
import numpy as np
import csv
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
    
    n_batches = loader.shape[0]
    pbar = tqdm(loader,
                colour='green',
                desc=f'Saving test labels',
                ncols=shutil.get_terminal_size().columns)

    # Initialize an empty list to store batch information
    
    # Initialize an empty list to store batch information
    batch_info_array = np.zeros((n_batches, int(batch_size)), dtype=np.int_)
    dataset_size = 0
    for batch_id, labels in enumerate(pbar):
        dataset_size += labels[labels != -1].shape[0]
        batch_info_array[batch_id, :labels.shape[0]] = labels
  
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
    
    fname = os.path.join(output_path,f'fault_classification_2.csv')
    
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
        loaded_faulty_output = np.load(faulty_output_filename).astype(np.float32)

        faulty_tensor_data = loaded_faulty_output
        n_faults = faulty_tensor_data.shape[0]
        n_classes = faulty_tensor_data.shape[-1]
        
        
        for z in tqdm(range(n_faults), desc="fault progress"):
            dim_batch =  batch_info_array[i].shape[0] if i < n_batches-1 else last_batch_size
            # Check numpy type and convert if necessary
            if faulty_tensor_data[z].dtype != np.float32:
                print(f'Converting faulty tensor data from {faulty_tensor_data[z].dtype} to float32')
                faulty_tensor_data[z] = faulty_tensor_data[z].astype(np.float32)
            if loaded_clean_output[i].dtype != np.float32:
                print(f'Converting clean output from {loaded_clean_output[i].dtype} to float32')
                loaded_clean_output[i] = loaded_clean_output[i].astype(np.float32)
            if batch_info_array[i].dtype != np.int_:
                print(f'Converting batch info from {batch_info_array[i].dtype} to int')
                batch_info_array[i] = batch_info_array[i].astype(np.int_)
                
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
for py_version in ['lib.linux-x86_64-cpython-310', 'lib.linux-x86_64-cpython-39']:
    full_path = os.path.join(build_dir, py_version)
    if os.path.exists(full_path) and full_path not in sys.path:
        sys.path.insert(0, full_path)
        break

#import cython_process_batch
#from cython_process_batch import process_batch_serial
import process_batch_serial
from process_batch_serial import process_a_batch, process_a_fault,process_a_fault_writing
    
import contextlib
from datetime import datetime
train = False
train_split = 1


batch_size = '64'
test_batch_size = int(batch_size)


REDUCED = False
TRIM_FOR_BER = True
DIVIDE_PER_LAYER = False
EMBEDDING = True
WRITE_IMAGES_RESPONSE = True
WRITE = False


dict_names_networks = {
    'Imagenette' : [
                'VGG11_IMAGENETTE',
                ],
}

dict_names_networks = {
    'CIFAR10' : [
                'AlexNet_static_ber_1e-6',
                'AlexNet_dynamic_EE3_ber_1e-4',
                'AlexNet_dynamic_EE3_ber_1e-5',
                'AlexNet_dynamic_EE3_ber_1e-6',
                'AlexNet_dynamic_EE5_ber_1e-4',
                'AlexNet_dynamic_EE5_ber_1e-5',
                'AlexNet_dynamic_EE5_ber_1e-6',
                'ResNet_dynamic_EE2_ber_1e-4',
                'ResNet_dynamic_EE2_ber_1e-5',
                'ResNet_dynamic_EE2_ber_1e-6',
                'ResNet_dynamic_EE3_ber_1e-4',
                'ResNet_dynamic_EE3_ber_1e-5',
                'ResNet_dynamic_EE3_ber_1e-6',
                'ResNet_dynamic_EE4_ber_1e-4',
                'ResNet_dynamic_EE4_ber_1e-5',
                'ResNet_dynamic_EE4_ber_1e-6',
                'ResNet_static_ber_1e-4',
                'ResNet_static_ber_1e-5',
                'ResNet_static_ber_1e-6',
                'VGG_dynamic_EE2_ber_1e-4',
                'VGG_dynamic_EE2_ber_1e-5',
                'VGG_dynamic_EE2_ber_1e-6',
                'VGG_dynamic_EE3_ber_1e-4',
                'VGG_dynamic_EE3_ber_1e-5',
                'VGG_dynamic_EE3_ber_1e-6',
                'VGG_dynamic_EE5_ber_1e-4',
                'VGG_dynamic_EE5_ber_1e-5',
                'VGG_dynamic_EE5_ber_1e-6',
                'VGG_static_ber_1e-4',
                'VGG_static_ber_1e-5',
                'VGG_static_ber_1e-6',
                ],
}

RANDOM_STATE = 51195


for DATASET in dict_names_networks.keys():
    for NETWORK in dict_names_networks[DATASET]:

        print(f'Analyzing {NETWORK} on {DATASET} with batch size {batch_size}')
        label_list = np.load(r"/home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI/data/cifar-10-batches-py/cifar10_test_labels.npy")

        clean_output_path = f'./output/clean_output/{DATASET}/{NETWORK}/batch_{batch_size}/clean_output.npy'
        faulty_output_path = f'./output/faulty_output/{DATASET}/{NETWORK}/batch_{batch_size}/stuck-at_params/{RANDOM_STATE}'
        output_path = '/home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI/OPT_FI_EXP/'
        out_dir = os.path.join(output_path, f'{NETWORK}_fault_classification' )
        if EMBEDDING:
            out_dir = os.path.join(out_dir, f'EMBEDDING', f'FAULT_LIST_RS_{RANDOM_STATE}')
        output_file = os.path.join(out_dir,f'{NETWORK}_fault_classification.out' if not REDUCED else f'reduced_fault_classification.out' )
        faults_indices_list = None
               
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        #masked, not_critical, critical, total, clean_acc, faulty_acc =  analyze_FI_output(test_loader, batch_size, clean_output_path, faulty_output_path,out_dir,WRITE_IMAGES_RESPONSE)
        analyze_FI_output(label_list, batch_size, clean_output_path, faulty_output_path,out_dir,WRITE_IMAGES_RESPONSE)

