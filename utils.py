import os
import argparse

import numpy as np
import pandas as pd
import shutil
from typing import Union, List, Tuple

import torch
from torch.nn import Sequential, Module
from torchvision.models.densenet import _DenseBlock, _Transition
from torchvision.models.efficientnet import Conv2dNormActivation
from torchvision.models import convnext_tiny 
from torchvision.models import ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from models import nicolo_net
from models.CIFAR10 import resnet_cifar10,mobilenetv2_cifar10 #, densenet_cifar10, googlenet_cifar10, , vgg_cifar10
# from models.CIFAR100 import densenet_cifar100, resnet_cifar100, resnext_cifar100, googlenet_cifar100
# from models.GTSRB import densenet_GTSRB, resnet_GTSRB, vgg_GTSRB



from FaultGenerators.FaultListGenerator import FaultListGenerator
from FaultGenerators.NeuronFault import NeuronFault
from FaultGenerators.WeightFault import WeightFault
from models.SmartLayers.SmartModulesManager import SmartModulesManager
from models.utils import load_from_dict, load_imagenet_datasets, Load_FMNIST_datasets, Load_MNIST_datasets, Load_CIFAR100_datasets, load_CIFAR10_datasets, load_CIFAR10_datasets_normalized, Load_GTSRB_datasets


import csv
from tqdm import tqdm



class UnknownNetworkException(Exception):
    pass


def parse_args():
    
    """
    Parse the argument of the network
    :return: The parsed argument of the network
    """

    parser = argparse.ArgumentParser(description = 'Run a fault injection campaign',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--force-n', type=int, default=None,
                        help='Force n fault injections')
    parser.add_argument('--forbid-cuda', action='store_true',
                        help='Completely disable the usage of CUDA. This command overrides any other gpu options.')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use the gpu if available.')
    parser.add_argument('--force-reload', action='store_true',
                        help='Force the computation of the output feature map.')
    parser.add_argument('--no-log-results', action='store_true',
                        help='Forbid logging the results of the fault injection campaigns')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='Test set batch size')
    parser.add_argument('--fault-model', '-m', type=str, required=True,
                        help='The fault model used for the fault injection',
                        choices=['byzantine_neuron', 'stuck-at_params'])
    parser.add_argument('--network-name', '-n', type=str,
                        help='Target network',
                        )
    parser.add_argument('--dataset', '-d', type=str, 
                        help='Dataset to use',
                        choices=['CIFAR10','CIFAR100','GTSRB','MNIST'])
    
    parser.add_argument('--dry-run', '-r', type=bool, 
                        help='Check if model and dataset are correctly loaded',
                        choices=[True,False])
    
    parser.add_argument('--train-set', action='store_true',
                        help='Run the fault injection campaign on the training set')
    
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='The threshold under which an error is undetected')
    parser.add_argument('--enable_gaussian_filter', action='store_true',
                        help='Apply the gaussian filter to the ofm to decrease fault impact')

    parsed_args = parser.parse_args()

    return parsed_args




def clean_inference(network, loader, device, network_name):
       

    clean_output_scores = list()
    clean_output_indices = list()
    clean_labels = list()

    counter = 0
    with torch.no_grad():
        
        pbar = tqdm(loader,
                colour='green',
                desc=f'Clean Run',
                ncols=shutil.get_terminal_size().columns)
    
        dataset_size = 0
        
        for batch_id, batch in enumerate(pbar):
            
            data, label = batch
            dataset_size = dataset_size + len(label)
            data = data.to(device)
            
            network_output = network(data)
            prediction = torch.topk(network_output, k=1)
            scores = network_output.cpu()
            indices = [int(fault) for fault in prediction.indices]
            
            clean_output_scores.append(scores.numpy())
            clean_output_indices.append(indices)
            clean_labels.append(label)
            
            counter = counter + 1


        elementwise_comparison = [label != index for labels, indices in zip(clean_labels, clean_output_indices) for label, index in zip(labels, indices)]          
        # Count the number of different elements
        num_different_elements = elementwise_comparison.count(True)
        
        print(f'device: {device}')
        print(f'network: {network_name}')
        print(f"The DNN wrong predicions are: {num_different_elements}")
        accuracy= (1 - num_different_elements/dataset_size)*100
        print(f"The final accuracy is: {accuracy}%")
        
        return clean_output_scores

     

def get_network(network_name: str,
                device: torch.device,
                dataset_name: str,
                root: str = '.') -> torch.nn.Module:
    """
    Load the network with the specified name
    :param network_name: The name of the network to load
    :param device: the device where to load the network
    :param root: the directory where to look for weights
    :return: The loaded network
    """
    if 'PRUNED' in network_name:
        return load_pruned_network(network_name, device, dataset_name, root)

    if dataset_name == 'CIFAR100':
        print(f'Loading network {network_name}')    
        if 'ResNet18' in network_name:  
            network = resnet_cifar100.resnet18()
            print('resnet18 loaded')
        elif 'DenseNet' in network_name:  
            network = densenet_cifar100.densenet121()
            print('densenet121 loaded')
        elif 'GoogLeNet' in network_name:
            network = googlenet_cifar100.googlenet()
            print('googlenet loaded')
        elif 'ResNext' in network_name:
            network = resnext_cifar100.resnext50()
            print('resnext50 loaded')
        elif 'HCGNet' in network_name:
            if 'A1' in network_name:
                network = nicolo_net.HCGNet_A1(num_classes=100).cuda()
            elif 'A2' in network_name:
                network = nicolo_net.HCGNet_A2(num_classes=100).cuda()
            else:
                network = nicolo_net.HCGNet_A3(num_classes=100).cuda()
        elif 'RESNET56' in network_name:
            network = nicolo_net.cifar_resnet56(num_classes=100)
            network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
            state_dict = torch.load(network_path, map_location=device, weights_only=True)
            network.load_state_dict(state_dict)
            network = nicolo_net.CifarResNet_sequential(network)
        elif 'VGG13' in network_name:
            network = nicolo_net.cifar_vgg13_bn(num_classes=100)
            network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
            state_dict = torch.load(network_path, map_location=device, weights_only=True)
            network.load_state_dict(state_dict)
            network = nicolo_net.vgg_sequential(network)  
        elif 'MOBILENETV2_X1_0' in network_name:
            network = nicolo_net.cifar_mobilenetv2_x1_0(num_classes=100)
            network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
            state_dict = torch.load(network_path, map_location=device, weights_only=True)
            network.load_state_dict(state_dict)
            network = nicolo_net.CifarMobileNetV2_sequential(network) 
        elif 'MOBILENETV2_X1_4' in network_name:
            network = nicolo_net.cifar_mobilenetv2_x1_4(num_classes=100)
            network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
            state_dict = torch.load(network_path, map_location=device, weights_only=True)
            network.load_state_dict(state_dict)
            network = nicolo_net.CifarMobileNetV2_sequential(network)  
        elif 'SHUFFLENETV2' in network_name:
            network = nicolo_net.cifar_shufflenetv2_x2_0(num_classes=100)
            network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
            state_dict = torch.load(network_path, map_location=device, weights_only=True)
            network.load_state_dict(state_dict)
            network = nicolo_net.CifarShuffleNetV2_sequential(network)
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of the model: {network_name}')

        # Load the weights
        if 'HCGNet' in network_name:
            network = torch.nn.DataParallel(network)
            network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pth.tar'
            state_dict = torch.load(network_path, map_location=device)
            network.load_state_dict(state_dict['net'], strict=True, weights_only=True)
            network = nicolo_net.HCGNet_sequential(network)
        elif 'RESNET56' in network_name:
            pass
        elif 'VGG13' in network_name:
            pass
        elif 'MOBILENETV2_X1' in network_name:
            pass
        elif 'MOBILENETV2_X1_4' in network_name:
            pass
        elif 'SHUFFLENETV2' in network_name:
            pass
        else:        
            network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pth'
            function = None
            
            state_dict = torch.load(network_path, map_location=device)['state_dict'] if '.th' in network_path else torch.load(network_path, map_location=device)
            clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()} if function is None else {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}
            network.load_state_dict(clean_state_dict, strict=False)
        
        network.to(device)
        network.eval()
        
    elif dataset_name == 'CIFAR10':
        network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'

        print(f'Loading network {network_name}')    
        if 'ResNet20' in network_name:  
            network = resnet_cifar10.resnet20()
            print('resnet20 loaded')
        elif 'ResNet32' in network_name:  
            network = resnet_cifar10.resnet32()
        elif 'ResNet44' in network_name:
            network = resnet_cifar10.resnet44()
        elif 'DenseNet121' in network_name:
            network = densenet_cifar10.densenet121()
        elif 'DenseNet161' in network_name:
            network = densenet_cifar10.densenet161()
        elif 'GoogLeNet' in network_name:
            network = googlenet_cifar10.googlenet()
        elif 'MobileNetV2' in network_name:
            network = mobilenetv2_cifar10.MobileNetV2()
            network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
            state_dict = torch.load(network_path, map_location=device)["net"]
            function = None
            if function is None:
                clean_state_dict = {
                    key.replace("module.", ""): value for key, value in state_dict.items()
                }
            else:
                clean_state_dict = {
                    key.replace("module.", ""): function(value)
                    if not (("bn" in key) and ("weight" in key))
                    else value
                    for key, value in state_dict.items()
                }

            network.load_state_dict(clean_state_dict, strict=False)
            network = mobilenetv2_cifar10.MobileNetV2_sequential(network)
        elif 'MOBILENET' in network_name:
            network = mobilenetv2_cifar10.MobileNetV2()
            network = mobilenetv2_cifar10.MobileNetV2_sequential(network)
            network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
            state_dict = torch.load(network_path, map_location=device)
            network.load_state_dict(state_dict, strict=True)
            
        elif 'LE_NET' in network_name or 'LENET' in network_name.upper():
            network = nicolo_net.le_net(pretrained=True,pretrained_path = network_path,channel_size=3)
        elif 'RESNET' in network_name:
            network = nicolo_net.resnet_sequential(nicolo_net.ResNet18(channel_size = 3), pretrained=True,pretrained_path = network_path)
        elif 'VGG' in network_name:
            if '11' in network_name:
                network = nicolo_net.vgg_sequential(nicolo_net.vgg11(in_channels=3,batchnorm = True),pretrained=True,pretrained_path = network_path)
            elif '16' in network_name:
                network = nicolo_net.vgg_sequential(nicolo_net.vgg16(in_channels=3,batchnorm = True),pretrained=True,pretrained_path = network_path)
        elif 'BAIDU' in network_name:
            network = nicolo_net.baidu_sequential(pretrained=True,pretrained_path = network_path,channel_size=3, batchnorm = True)        
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of the model: {network_name}')
        
        print(network_name)

        # Load the weights
        if ('MobileNetV2' not in network_name):
            if 'ResNet' in network_name:
                network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.th'
            else:
                network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
        
            load_from_dict(network=network,
                            device=device,
                            path=network_path)
        
        network.to(device)
        network.eval()
        

        
    elif dataset_name == 'GTSRB':
        print(f'Loading network {network_name}')    
        if 'ResNet' in network_name:  
            network = resnet_GTSRB.resnet20()
        elif 'DenseNet' in network_name:  
            network = densenet_GTSRB.densenet121()
        elif 'VGG' in network_name:
            network = vgg_GTSRB.vgg11_bn()
        else:
            raise UnknownNetworkException(f'ERROR: unknown version of the model: {network_name}')

        # Load the weights
        network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
        
        load_from_dict(network=network,
                        device=device,
                        path=network_path)
        
        network.to(device)
        network.eval()
    
    elif dataset_name == 'MNIST':
    
        network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
        if 'MOBILENET' in network_name:
            network = mobilenetv2_cifar10.MobileNetV2(channel_size=1)
            network = mobilenetv2_cifar10.MobileNetV2_sequential(network)
            network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
            state_dict = torch.load(network_path, map_location=device)
            network.load_state_dict(state_dict, strict=True)
        elif 'LE_NET' in network_name or 'LENET' in network_name.upper():
            network = nicolo_net.le_net(pretrained=True,pretrained_path = network_path,channel_size=1)
        elif 'RESNET' in network_name:
            network = nicolo_net.resnet_sequential(nicolo_net.ResNet18(channel_size = 1), pretrained=True,pretrained_path = network_path)
        elif 'VGG' in network_name:
            if '11' in network_name:
                network = nicolo_net.vgg_sequential(nicolo_net.vgg11(in_channels=1,batchnorm = True),pretrained=True,pretrained_path = network_path)
            elif '16' in network_name:
                network = nicolo_net.vgg_sequential(nicolo_net.vgg16(in_channels=1,batchnorm = True),pretrained=True,pretrained_path = network_path)
        elif 'BAIDU' in network_name:
            network = nicolo_net.baidu_sequential(pretrained=True,pretrained_path = network_path,channel_size=1, batchnorm = True)        
            print(network_name)
            
        load_from_dict(network=network,
                            device=device,
                            path=network_path)
        network.to(device)
        network.eval()
    
    elif dataset_name == 'FMNIST':
    
        network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
        if 'MOBILENET' in network_name:
            network = mobilenetv2_cifar10.MobileNetV2(channel_size=1)
            network = mobilenetv2_cifar10.MobileNetV2_sequential(network)
            network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
            state_dict = torch.load(network_path, map_location=device)
            network.load_state_dict(state_dict, strict=True)
        elif 'LE_NET' in network_name or 'LENET' in network_name.upper():
            network = nicolo_net.le_net(pretrained=True,pretrained_path = network_path,channel_size=1)
        elif 'RESNET' in network_name:
            network = nicolo_net.resnet_sequential(nicolo_net.ResNet18(channel_size = 1), pretrained=True,pretrained_path = network_path)
        elif 'VGG' in network_name:
            if '11' in network_name:
                network = nicolo_net.vgg_sequential(nicolo_net.vgg11(in_channels=1,batchnorm = True),pretrained=True,pretrained_path = network_path)
            elif '16' in network_name:
                network = nicolo_net.vgg_sequential(nicolo_net.vgg16(in_channels=1,batchnorm = True),pretrained=True,pretrained_path = network_path)
        elif 'BAIDU' in network_name:
            network = nicolo_net.baidu_sequential(pretrained=True,pretrained_path = network_path,channel_size=1, batchnorm = True)        
            print(network_name)
            
        load_from_dict(network=network,
                            device=device,
                            path=network_path)
        network.to(device)
        network.eval()
    elif dataset_name == 'ImageNet':
        print(f'Loading network {network_name}')
        # By now support ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        if 'ConvNeXt_Tiny' in network_name:
            network = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            network.eval()
            network.to(device)
    else:
        raise UnknownNetworkException(f'ERROR: unknown network: {network_name}')

    # Send network to device and set for inference
   
    

    return network


def get_loader(network_name: str,
               batch_size: int,
               image_per_class: int = None,
               dataset_name: str = None,
               network: torch.nn.Module = None,
               train_split = 1,
               root='.') -> DataLoader:
    """
    Return the loader corresponding to a given network and with a specific batch size
    :param network_name: The name of the network
    :param batch_size: The batch size
    :param image_per_class: How many images to load for each class
    :param network: Default None. The network used to select the image per class. If not None, select the image_per_class
    that maximize this network accuracy. If not specified, images are selected at random
    :return: The DataLoader
    """
    if 'CIFAR10' == dataset_name:
        print('Network:', network_name)
        if network_name == 'MobileNetV2':
            train_loader, _, loader = load_CIFAR10_datasets_normalized(train_split=train_split,test_batch_size=batch_size,train_batch_size=batch_size,
                                             test_image_per_class=image_per_class,root=root)
        else:
            print('Loading CIFAR10 dataset')
            train_loader, _, loader = load_CIFAR10_datasets(train_split=train_split,test_batch_size=batch_size,train_batch_size=batch_size,
                                             test_image_per_class=image_per_class,root=root)
        
    elif 'CIFAR100' == dataset_name:
        print('Loading CIFAR100 dataset')
        train_loader, _, loader = Load_CIFAR100_datasets(train_split=train_split,test_batch_size=batch_size,train_batch_size=batch_size,
                                             test_image_per_class=image_per_class,root=root)
        
    elif 'GTSRB' == dataset_name:
        print('Loading GTSRB dataset')
        train_loader, _, loader = Load_GTSRB_datasets(train_split=train_split,test_batch_size=batch_size,train_batch_size=batch_size,
                                             test_image_per_class=image_per_class)
    
    elif 'MNIST' == dataset_name:
        print('Loading MNIST dataset')
        train_loader, _, loader = Load_MNIST_datasets(train_split=train_split,test_batch_size=batch_size,train_batch_size=batch_size,
                                             test_image_per_class=image_per_class,root=root)
    elif 'FMNIST' == dataset_name:
        print('Loading Fashion_MNIST dataset')
        train_loader, _, loader = Load_FMNIST_datasets(train_split=train_split,test_batch_size=batch_size,train_batch_size=batch_size,
                                             test_image_per_class=image_per_class,root=root)
    elif 'ImageNet' == dataset_name:
        print('Loading ImageNet dataset')
        train_loader, _, loader = load_imagenet_datasets(train_split=train_split,test_batch_size=batch_size,train_batch_size=batch_size,
                                             test_image_per_class=image_per_class,root=root,network_name = network_name)
    else:
        print('no dataset specified')
        exit()

    print(f'Batch size:\t\t{batch_size} \nNumber of batches:\t{len(loader)}')

    return train_loader, loader


def get_delayed_start_module(network: Module,
                             network_name: str) -> Module:
    """
    Get the delayed_start_module of the given network
    :param network: The instance of the network where to look for the fault_delayed_start module
    :param network_name: The name of the network
    :return: An instance of the delayed_start_module
    """

    # The module to change is dependent on the network. This is the module for which to enable delayed start
    if 'LeNet' in network_name:
        delayed_start_module = network
    elif 'ResNet' in network_name:
        delayed_start_module = network
    elif 'MobileNetV2' in network_name:
        delayed_start_module = network.features
        print('delayed_start_module:', delayed_start_module)
    elif 'DenseNet' in network_name:
        delayed_start_module = network.features
    elif 'EfficientNet' in network_name:
        delayed_start_module = network.features
    else:
        raise UnknownNetworkException

    return delayed_start_module


def get_module_classes(network_name: str) -> Union[List[type], type]:
    """
    Get the module_classes of a given network. The module classes represent the classes that can be replaced by smart
    modules in the network. Notice that the instances of these classes that will be replaced are only the children of
    the delayed_start_module
    :param network_name: The name of the network
    :return: The type of modules (or of a single module) that will should be replaced by smart modules in the target
    network
    """
    
    if 'ResNet' in network_name:
        if network_name in ['ResNet18', 'ResNet50']:
            module_classes = Sequential
        # else:
        #     module_classes = models.resnet.BasicBlock
    elif 'DenseNet' in network_name:
        module_classes = (_DenseBlock, _Transition)
    elif 'EfficientNet' in network_name:
        module_classes = (Conv2dNormActivation, Conv2dNormActivation)
    else:
        raise UnknownNetworkException(f'Unknown network {network_name}')

    return module_classes


def get_fault_list(fault_model: str,
                   fault_list_generator: FaultListGenerator,
                   seed: int = 42,
                   e: float = .01,
                   t: float = 2.58) -> Tuple[Union[List[NeuronFault], List[WeightFault]], List[Module]]:
    """
    Get the fault list corresponding to the specific fault model, using the fault list generator passed as argument
    :param fault_model: The name of the fault model
    :param fault_list_generator: An instance of the fault generator
    :param e: The desired error margin
    :param t: The t related to the desired confidence level
    :return: A tuple of fault_list, injectable_modules. The latter is a list of all the modules that can be injected in
    case of neuron fault injections
    """
    if fault_model == 'byzantine_neuron':
        fault_list = fault_list_generator.get_neuron_fault_list(load_fault_list=False,
                                                                save_fault_list=True,
                                                                e=e,
                                                                t=t)
    elif fault_model == 'stuck-at_params':
        fault_list = fault_list_generator.get_weight_fault_list(load_fault_list=True,
                                                                save_fault_list=False,
                                                                seed = seed,
                                                                e=e,
                                                                t=t)
    else:
        raise ValueError(f'Invalid fault model {fault_model}')

    injectable_modules = fault_list_generator.injectable_output_modules_list

    return fault_list, injectable_modules


def get_device(forbid_cuda: bool,
               use_cuda: bool) -> torch.device:
    """
    Get the device where to perform the fault injection
    :param forbid_cuda: Forbids the usage of cuda. Overrides use_cuda
    :param use_cuda: Whether to use the cuda device or the cpu
    :return: The device where to perform the fault injection
    """

    # Disable gpu if set
    if forbid_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = 'cpu'
        if use_cuda:
            print('WARNING: cuda forcibly disabled even if set_cuda is set')
    # Otherwise, use the appropriate device
    else:
        if use_cuda:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = ''
                print('ERROR: cuda not available even if use-cuda is set')
                exit(-1)
        else:
            device = 'cpu'

    return torch.device(device)


def formatted_print(fault_list: list,
                    network_name: str,
                    batch_size: int,
                    batch_id: int,
                    faulty_prediction_dict: dict,
                    fault_dropping: bool = False,
                    fault_delayed_start: bool = False) -> None:
    """
    A function that prints to csv the results of the fault injection campaign on a single batch
    :param fault_list: A list of the faults
    :param network_name: The name of the network
    :param batch_size: The size of the batch of the data loader
    :param batch_id: The id of the batch
    :param faulty_prediction_dict: A dictionary where the key is the fault index and the value is a list of all the
    top_1 prediction for all the image of the given the batch
    :param fault_dropping: Whether fault dropping is used or not
    :param fault_delayed_start: Whether fault delayed start is used or not
    """

    fault_list_rows = [[fault_id,
                       fault.layer_name,
                        fault.tensor_index[0],
                        fault.tensor_index[1] if len(fault.tensor_index) > 1 else np.nan,
                        fault.tensor_index[2] if len(fault.tensor_index) > 2 else np.nan,
                        fault.tensor_index[3] if len(fault.tensor_index) > 3 else np.nan,
                        fault.bit,
                        fault.value
                        ]
                       for fault_id, fault in enumerate(fault_list)
                       ]

    fault_list_columns = [
        'Fault_ID',
        'Fault_Layer',
        'Fault_Index_0',
        'Fault_Index_1',
        'Fault_Index_2',
        'Fault_Index_3',
        'Fault_Bit',
        'Fault_Value'
    ]

    prediction_rows = [
        [
            fault_id,
            batch_id,
            prediction_id,
            prediction[0],
            prediction[1],
        ]
        for fault_id in faulty_prediction_dict for prediction_id, prediction in enumerate(faulty_prediction_dict[fault_id])
    ]

    prediction_columns = [
        'Fault_ID',
        'Batch_ID',
        'Image_ID',
        'Top_1',
        'Top_Score',
    ]

    fault_list_df = pd.DataFrame(fault_list_rows, columns=fault_list_columns)
    prediction_df = pd.DataFrame(prediction_rows, columns=prediction_columns)

    complete_df = fault_list_df.merge(prediction_df, on='Fault_ID')

    file_prefix = 'combined_' if fault_dropping and fault_delayed_start \
        else 'delayed_' if fault_delayed_start \
        else 'dropping_' if fault_dropping \
        else ''

    output_folder = f'output/fault_campaign_results/{network_name}/{batch_size}'
    os.makedirs(output_folder, exist_ok=True)
    complete_df.to_csv(f'{output_folder}/{file_prefix}fault_injection_batch_{batch_id}.csv', index=False)


def enable_optimizations(
        network: Module,
        delayed_start_module: Union[Module, None],
        module_classes: Union[List[type], type],
        device: torch.device,
        fm_folder: str,
        fault_list_generator: FaultListGenerator,
        fault_list: Union[List[NeuronFault], List[WeightFault]],
        input_size: torch.Size = torch.Size((1, 3, 32, 32)),
        injectable_modules: List[Module] = None,
        fault_delayed_start: bool = True,
        fault_dropping: bool = True):

    # Replace the convolutional layers
    if fault_dropping or fault_delayed_start:

        smart_layers_manager = SmartModulesManager(network=network,
                                                   delayed_start_module=delayed_start_module,
                                                   device=device,
                                                   input_size=input_size)

        if fault_delayed_start:
            # Replace the forward module of the target module to enable delayed start
            smart_layers_manager.replace_module_forward()

        # Replace the smart layers of the network
        smart_modules_list = smart_layers_manager.replace_smart_modules(module_classes=module_classes,
                                                                        fm_folder=fm_folder,
                                                                        fault_list=fault_list)

        # Update the network. Useful to update the list of injectable layers when injecting in the neurons
        if injectable_modules is not None:
            fault_list_generator.update_network(network)
            injectable_modules = fault_list_generator.injectable_output_modules_list

        network.eval()
    else:
        smart_modules_list = None

    return injectable_modules, smart_modules_list


def load_pruned_network(network_name, device, dataset_name, root='.'):
    
    """
    Load a pruned network from a checkpoint file containing the state dict,
    the pruned neuron indices, the original network name and the dataset name.
    Reconstructs the pruned architecture and loads the saved weights.

    """

    import sys
    sys.path.append('/home/nicolo_b/Desktop/PhD/RELIABLE_NAS/NOTEBOOK/FAULT_INJECTOR/VITTFI/OPT_FI_EXP/ACTIVATION_BASED_PRUNING')
    from utility import create_column_selector, prune_neurons

    network_path = f'{root}/models/{dataset_name}/pretrained/{network_name}_{dataset_name}.pt'
    print(f'Loading pruned network from {network_path}')
    
    checkpoint = torch.load(network_path, map_location=device)
    
    original_name  = checkpoint['original_network']
    pruned_indices = checkpoint['pruned_indices']
    
    print(f'Original network: {original_name}')
    print(f'Pruned neurons: {len(pruned_indices)}')
    
    network = get_network(network_name=original_name,
                          device=device,
                          dataset_name=dataset_name,
                          root=root)
    
    zero_indices = [i for i in range(network[-1].in_features)
                    if i not in pruned_indices]

    selection_layer = create_column_selector(
        network[-1].in_features,
        len(pruned_indices),
        pruned_indices
    )

    selection_layer = selection_layer.to(device)
    
    network = prune_neurons(network, zero_indices)

    network.insert(-1, selection_layer)
    network = network.to('cpu')
    
    network.load_state_dict(checkpoint['state_dict'])
    network.to(device)
    network.eval()
    
    print('Pruned network loaded successfully!')
    return network


