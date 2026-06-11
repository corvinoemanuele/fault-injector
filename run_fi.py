import copy
import os
import sys
from tqdm import tqdm
import gc

import torch
from torch import nn

from FaultGenerators.FaultListGenerator import FaultListGenerator
from FaultInjectionManager import FaultInjectionManager
from OutputFeatureMapsManager.OutputFeatureMapsManager import OutputFeatureMapsManager
from utils import get_network, get_device, parse_args, get_loader, get_module_classes, \
    get_delayed_start_module, enable_optimizations, get_fault_list, clean_inference

from config import (
    NETWORK_NAME, DATASET, BATCH_SIZE, DEVICE, ROOT, DATASET_ROOT,
    FAULT_MODEL, SEED, PRUNING, TMR, MAX_FAULTS_TO_INJECT,
    FORCE_RELOAD, TRAIN, DRY_RUN, PRINT,
    SEED_IMAGENET, FORBID_CUDA, OUTPUTFMANALYZER, DEBUG
)

old_dir = os.getcwd()
print(old_dir)
utility_dir = f'{ROOT}/OPT_FI_EXP'
tmr_dir = f'{ROOT}/OPT_FI_EXP/TMR'

sys.path.append(utility_dir)

from analyze_utility_selection import prune_network,embedding_images_given_net
sys.path.remove(utility_dir)
os.chdir(old_dir)
sys.path.append(tmr_dir)
from mymodels import ReducedTMRModel,TMRModel




def main():
    global NETWORK_NAME
    # Set deterministic algorithms
    torch.use_deterministic_algorithms(mode=True,warn_only=True)
    
    # Select the device
    device = get_device(forbid_cuda=FORBID_CUDA,
                        use_cuda=DEVICE)
    
    print(f'Using device {device}')
    
    # Load the dataset
    if DATASET == 'ImageNet':
        train_loader, loader = get_loader(network_name=NETWORK_NAME,
                            batch_size=BATCH_SIZE,
                            dataset_name=DATASET,
                            root=DATASET_ROOT,
                            train_split = SEED_IMAGENET #THIS IS USED AS SEED FOR RANDOM IMAGE SAMPLING
                            )
    else:
        train_loader, loader = get_loader(network_name=NETWORK_NAME,
                            batch_size=BATCH_SIZE,
                            dataset_name=DATASET,
                            root=DATASET_ROOT,
                            )
    
    
    if TRAIN:
        loader = train_loader
        
    # Load the network
    network = get_network(network_name=NETWORK_NAME,
                          device=device,
                          dataset_name=DATASET,
                          root=ROOT)
    
    if PRUNING:
        print('pruning network...')
        
        copied_network = copy.deepcopy(network)
        copied_network = prune_network(NETWORK_NAME, copied_network, train_loader, device, ROOT, DATASET)
        
        copied_network.to(device)
        network.to('cpu')
        del network
        network = copied_network
        gc.collect()
        NETWORK_NAME = NETWORK_NAME + '_PRUNED'
        print('pruning completed!')
    
    
    
    if TMR:
        print('Enabling TMR...')
        NETWORK_NAME = NETWORK_NAME + '_TMR'
        network_path = os.path.join(tmr_dir,f'{NETWORK_NAME}_{DATASET}.pt')
        copied_network = copy.deepcopy(network)
        network = TMRModel(copied_network.to('cpu'),dropout_rate=0.25).to(device)
        network.load_state_dict(torch.load(network_path))
        print('TMR enabled!')
    
    
    network.eval()
    print('clean inference accuracy test:')
    
    if OUTPUTFMANALYZER:
        print('Output Feature Maps Analyzer')
        NETWORK_NAME = NETWORK_NAME + '_OFM'
        network = nn.Sequential(*list(network.children())[:-1])
        network.eval()
        network.to(device)
        
        
    with torch.no_grad():
        clean_output = clean_inference(network, loader, device, NETWORK_NAME)
        
        print(f'clean output: {clean_output[0].shape}')
    
        # Folder containing the feature maps
        # clean_fm_folder = f'output/clean_feature_maps/{args.network_name}/batch_{args.batch_size}'
        # faulty_fm_folder = f'output/faulty_feature_maps/{args.network_name}/batch_{args.batch_size}/{args.fault_model}'

        clean_fm_folder = f'./output/clean_feature_maps/{DATASET}/{NETWORK_NAME}/batch_{BATCH_SIZE}'
        faulty_fm_folder = f'./output/faulty_feature_maps/{DATASET}/{NETWORK_NAME}/batch_{BATCH_SIZE}/{FAULT_MODEL}/{SEED}'
        os.makedirs(clean_fm_folder, exist_ok=True)
        os.makedirs(faulty_fm_folder, exist_ok=True)

        
        # Folder containing the clean output
        # clean_output_folder = f'output/clean_output/{args.network_name}/batch_{args.batch_size}'
        clean_output_folder = f'./output/clean_output/{DATASET}/{NETWORK_NAME}/batch_{BATCH_SIZE}'

        #attenzione a module_classes che mi salva ofm diverse!
        module_classes = (torch.nn.Conv2d, torch.nn.Linear)
        

        
        feature_maps_layer_names = [name.replace('.weight', '') for name, module in network.named_modules()
                                            if isinstance(module, module_classes)]
        
        print('feature maps layer names:')
        print(feature_maps_layer_names)
        
        
        clean_ofm_manager = OutputFeatureMapsManager(network=network,
                                                    loader=loader,
                                                    module_classes=module_classes,
                                                    device=device,
                                                    fm_folder=clean_fm_folder,
                                                    clean_output_folder=clean_output_folder)

        # Try to load the clean input
        clean_ofm_manager.load_clean_output(force_reload=FORCE_RELOAD)

        
        # Generate fault list
        fault_list_generator = FaultListGenerator(network=network,
                                                network_name=NETWORK_NAME,
                                                device=device,
                                                module_class=torch.nn.Conv2d,
                                                input_size=loader.dataset[0][0].unsqueeze(0).shape,
                                                save_ifm=True)


        
        # Create a smart network. a copy of the network with its convolutional layers replaced by their smart counterpart
        smart_network = copy.deepcopy(network)
        fault_list_generator.update_network(smart_network)

        print('fault list preparation:')
        # Manage the fault models
        fault_list, injectable_modules = get_fault_list(fault_model=FAULT_MODEL,
                                                        fault_list_generator=fault_list_generator,
                                                        seed=SEED)

        if DEBUG:
            print('injectable_modules:')
            from pprint import pprint
            pprint(injectable_modules)
            pprint(len(injectable_modules))
            for module in network.modules():
                if isinstance(module, torch.nn.Conv2d):
                    print('conv2d')
                    print(module)
                    print(module.weight.shape)
                elif isinstance(module, torch.nn.Linear):
                    print('linear')
                    print(module)
                    print(module.weight.shape)

        faulty_output_folder = f'./output/faulty_output/{DATASET}/{NETWORK_NAME}/batch_{BATCH_SIZE}/{FAULT_MODEL}/{SEED}'
        log_folder = f'log/{DATASET}/{NETWORK_NAME}/batch_{BATCH_SIZE}/{FAULT_MODEL}/{SEED}'

        # Execute the fault injection campaign with the smart network
        fault_injection_executor = FaultInjectionManager(network=smart_network,
                                                        network_name=NETWORK_NAME,
                                                        device=device,
                                                        smart_modules_list=None,
                                                        loader=loader,
                                                        dataset_name=DATASET,
                                                        clean_output=clean_ofm_manager.clean_output,
                                                        injectable_modules=injectable_modules,
                                                        faulty_output_folder=faulty_output_folder,
                                                        log_folder=log_folder)
        
        
        
        if DRY_RUN:
            print('Dry run completed')
            exit(0)
            
        fault_injection_executor.run_faulty_campaign_on_weight(fault_model=FAULT_MODEL,
                                                            fault_list=fault_list,
                                                            fault_dropping=False,
                                                            fault_delayed_start=False,
                                                            delayed_start_module=None,
                                                            first_batch_only=False,
                                                            force_n=MAX_FAULTS_TO_INJECT,
                                                            save_output=True,
                                                            save_ofm=False,
                                                            ofm_folder=faulty_fm_folder)



main()