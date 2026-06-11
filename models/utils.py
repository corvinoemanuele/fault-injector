import os
import pickle
import numpy as np

import torch
import math
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, GTSRB, MNIST, FashionMNIST, ImageNet
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision.transforms.v2 import ToTensor,Resize,Compose,ColorJitter,RandomRotation,AugMix,RandomCrop,GaussianBlur,RandomEqualize,RandomHorizontalFlip,RandomVerticalFlip

from torchvision.models import convnext_tiny
from torchvision.models import ConvNeXt_Tiny_Weights


def load_unet_dataset(batch_size=32):

    filename = 'weights/unet_loader.npy'

    with open(filename, 'rb') as f:
        dataset_list = pickle.load(f)

        dataset_x = torch.stack([tensor.squeeze() for tensor in list(zip(*dataset_list))[0]])
        dataset_y = torch.stack([tensor.squeeze().int() for tensor in list(zip(*dataset_list))[1]])

        dataset_x = dataset_x[[bool(tensor.sum() == 0) for tensor in dataset_y]]
        dataset_y = dataset_y[[bool(tensor.sum() == 0) for tensor in dataset_y]]

        dataset = torch.utils.data.TensorDataset(dataset_x, dataset_y)

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return loader


def load_CIFAR10_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None,root='.'):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                       # Crop the image to 32x32
        transforms.RandomHorizontalFlip(),                                          # Data Augmentation
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),                                                  # Crop the image to 32x32
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    path = os.path.join(root,'data')
    train_dataset = CIFAR10(path,
                            train=True,
                            transform=transform_train,
                            download=True)
    test_dataset = CIFAR10(path,
                           train=False,
                           transform=transform_test,
                           download=True)

    if test_image_per_class is not None:
        selected_test_list = []
        image_class_counter = [0] * 10
        for test_image in test_dataset:
            if image_class_counter[test_image[1]] < test_image_per_class:
                selected_test_list.append(test_image)
                image_class_counter[test_image[1]] += 1
        test_dataset = selected_test_list

    # Split the training set into training and validation
    if train_split == 1:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=train_batch_size,
                                                   shuffle=False)
        val_loader = None
    else:
        train_split_length = int(len(train_dataset) * train_split)
        val_split_length = len(train_dataset) - train_split_length
        train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                                lengths=[train_split_length, val_split_length],
                                                                generator=torch.Generator().manual_seed(1234))
        # DataLoader is used to load the dataset
        # for training
        train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                                batch_size=train_batch_size,
                                                shuffle=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                                batch_size=train_batch_size,
                                                shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    print('Dataset loaded')

    return train_loader, val_loader, test_loader

def load_CIFAR10_datasets_normalized(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None,root = '.'):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                       # Crop the image to 32x32
        transforms.RandomHorizontalFlip(),                                          # Data Augmentation
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),                                                  # Crop the image to 32x32
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    path = os.path.join(root,'data')
    train_dataset = CIFAR10(path,
                            train=True,
                            transform=transform_train,
                            download=True)
    test_dataset = CIFAR10(path,
                           train=False,
                           transform=transform_test,
                           download=True)

    if test_image_per_class is not None:
        selected_test_list = []
        image_class_counter = [0] * 10
        for test_image in test_dataset:
            if image_class_counter[test_image[1]] < test_image_per_class:
                selected_test_list.append(test_image)
                image_class_counter[test_image[1]] += 1
        test_dataset = selected_test_list

    # Split the training set into training and validation
    if train_split == 1:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=train_batch_size,
                                                   shuffle=False)
        val_loader = None
    else:
        train_split_length = int(len(train_dataset) * train_split)
        val_split_length = len(train_dataset) - train_split_length
        train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                                lengths=[train_split_length, val_split_length],
                                                                generator=torch.Generator().manual_seed(1234))
        # DataLoader is used to load the dataset
        # for training
        train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                                batch_size=train_batch_size,
                                                shuffle=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                                batch_size=train_batch_size,
                                                shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    print('Dataset loaded')

    return train_loader, val_loader, test_loader

def Load_MNIST_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None,root='.'):

    transform_train = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomAffine(degrees=20,translate=(0.1,0.1),scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the data (stability for training)
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the data (stability for training)
        ])
    path = os.path.join(root,'data')
    train_dataset = MNIST(path,
                            train=True,
                            transform=transform_train,
                            download=True)
    test_dataset = MNIST(path,
                           train=False,
                           transform=transform_test,
                           download=True)

    if test_image_per_class is not None:
        selected_test_list = []
        image_class_counter = [0] * 10
        for test_image in test_dataset:
            if image_class_counter[test_image[1]] < test_image_per_class:
                selected_test_list.append(test_image)
                image_class_counter[test_image[1]] += 1
        test_dataset = selected_test_list

    # Split the training set into training and validation
    if train_split == 1:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=train_batch_size,
                                                   shuffle=False)
        val_loader = None
    else:
        train_split_length = int(len(train_dataset) * train_split)
        val_split_length = len(train_dataset) - train_split_length
        train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                                lengths=[train_split_length, val_split_length],
                                                                generator=torch.Generator().manual_seed(1234))
        # DataLoader is used to load the dataset
        # for training
        train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                                batch_size=train_batch_size,
                                                shuffle=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                                batch_size=train_batch_size,
                                                shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    print('Dataset loaded')

    return train_loader, val_loader, test_loader

def load_imagenet_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None,root='.', network_name=None):

    if 'ConvNeXt_Tiny' in network_name:
        transform_train = transform_test = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    path = os.path.join(root,'ImageNet')
    
    # THIS IS A TEMPORARY FIX, THE IMAGENET DATASET IS NOT AVAILABLE IN THE NOTEBOOK
    
    train_loader = val_loader = test_loader = None
    
    # Here i am taking a subset of the ImageNet dataset for testing purposes
    seed = int(train_split*1000)
    
    dataset = ImageNet(path,
                        split='val',
                        transform=transform_test,
                        )
    n_images_to_sample = 4000
    print(f"Sampling {n_images_to_sample} images from ImageNet, seed {seed}")
    random_indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:n_images_to_sample]
    subset = Subset(dataset, random_indices)
    test_loader = torch.utils.data.DataLoader(
        dataset=subset,
        batch_size=test_batch_size,
        shuffle=False
    )
    return train_loader, val_loader, test_loader

def Load_FMNIST_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None,root='.'):

    transform_train = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomAffine(degrees=20,translate=(0.1,0.1),scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the data (stability for training)
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the data (stability for training)
        ])
    path = os.path.join(root,'data')
    train_dataset = FashionMNIST(path,
                            train=True,
                            transform=transform_train,
                            download=True)
    test_dataset = FashionMNIST(path,
                           train=False,
                           transform=transform_test,
                           download=True)

    if test_image_per_class is not None:
        selected_test_list = []
        image_class_counter = [0] * 10
        for test_image in test_dataset:
            if image_class_counter[test_image[1]] < test_image_per_class:
                selected_test_list.append(test_image)
                image_class_counter[test_image[1]] += 1
        test_dataset = selected_test_list

    # Split the training set into training and validation
    if train_split == 1:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=train_batch_size,
                                                   shuffle=False)
        val_loader = None
    else:
        train_split_length = int(len(train_dataset) * train_split)
        val_split_length = len(train_dataset) - train_split_length
        train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                                lengths=[train_split_length, val_split_length],
                                                                generator=torch.Generator().manual_seed(1234))
        # DataLoader is used to load the dataset
        # for training
        train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                                batch_size=train_batch_size,
                                                shuffle=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                                batch_size=train_batch_size,
                                                shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    print('Dataset loaded')

    return train_loader, val_loader, test_loader


def Load_CIFAR100_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None,root='.'):
    
    transform_train = transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomRotation(15),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                     [0.2675, 0.2565, 0.2761])
                                             ])
    
    transform_test = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                     [0.2675, 0.2565, 0.2761]),
                                            ])

    
    path = root
    train_dataset = CIFAR100(path,
                            train=True,
                            transform=transform_train,
                            download=False)
    test_dataset = CIFAR100(path,
                           train=False,
                           transform=transform_test,
                           download=False)

    if test_image_per_class is not None:
        selected_test_list = []
        image_class_counter = [0] * 10
        for test_image in test_dataset:
            if image_class_counter[test_image[1]] < test_image_per_class:
                selected_test_list.append(test_image)
                image_class_counter[test_image[1]] += 1
        test_dataset = selected_test_list

    # Split the training set into training and validation
    if train_split == 1:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=train_batch_size,
                                                   shuffle=False)
        val_loader = None
    else:
        train_split_length = int(len(train_dataset) * train_split)
        val_split_length = len(train_dataset) - train_split_length
        train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                                lengths=[train_split_length, val_split_length],
                                                                generator=torch.Generator().manual_seed(1234))
        # DataLoader is used to load the dataset
        # for training
        train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                                batch_size=train_batch_size,
                                                shuffle=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                                batch_size=train_batch_size,
                                                shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    print('Dataset loaded')

    return train_loader, val_loader, test_loader


def Load_GTSRB_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):
    
    train_transforms = Compose([
    ColorJitter(brightness=1.0, contrast=0.5, saturation=1, hue=0.1),
    RandomEqualize(0.4),
    AugMix(),
    RandomHorizontalFlip(0.3),
    RandomVerticalFlip(0.3),
    GaussianBlur((3,3)),
    RandomRotation(30),
    
    Resize([50,50]),
    ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                            (0.2724, 0.2608, 0.2669))
    
    ])

    validation_transforms = Compose([
        Resize([50, 50]),
        ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    ])

    train_dataset = GTSRB(root='./data',
                            split='train',
                            download=True,
                            transform=train_transforms)
    test_dataset = GTSRB(root='./data',
                            split='test',
                            download=True,
                            transform=validation_transforms)



    # Split the training set into training and validation
    train_split_length = int(len(train_dataset) * 0.8)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                                lengths=[train_split_length, val_split_length],
                                                                generator=torch.Generator().manual_seed(1234))
    # DataLoader is used to load the dataset
    # for training
    train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                                batch_size=train_batch_size,
                                                shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                                batch_size=train_batch_size,
                                                shuffle=True)  

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=test_batch_size,
                                                shuffle=False)

    print('GTSRB Dataset loaded')
        
    return train_loader, val_loader, test_loader

def load_from_dict(network, device, path, function=None):
    
    if '.th' in path:
        state_dict = torch.load(path, map_location=device)['state_dict']
        print('state dict loaded')
    else:
        state_dict = torch.load(path, map_location=device)
        print('state dict loaded')

    if function is None:
        clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    else:
        clean_state_dict = {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}

    network.load_state_dict(clean_state_dict, strict=False)
    print('Weights loaded from disk')
    
    
def get_module_by_name():
    pass
