from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from backpack.custom_module.branching import Parallel, SumModule
import math

__all__ = [
    'VGG', 'vgg11', 'vgg13',  'vgg16', 'vgg19', 'ResNet18', 'resnet_sequential', 'vgg_sequential', 'baidu_sequential', 'le_net'
]




def resnet9(pretrained = False, pretrained_path = None, **kwargs):
    """Constructs a ResNet-9 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on CIFAR10
    """
    channel_size = kwargs.get('channel_size',1)
    dropout = kwargs.get('dropout',0.25)
    
    model = model = nn.Sequential(
                nn.Conv2d(channel_size, 64, kernel_size=3, stride=1, padding=1, bias=False),
                #nn.BatchNorm2d(64),
                nn.ReLU(),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                #nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                Parallel(
                    nn.Identity(),
                    nn.Sequential(
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False),
                        #nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False),
                        #nn.BatchNorm2d(128),
                        nn.ReLU()),
                    merge_module=SumModule()
                ),
                
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                #nn.BatchNorm2d(256),
                nn.ReLU(),
                
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                #nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                Parallel(
                    nn.Identity(),
                    nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
                        #nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
                        #nn.BatchNorm2d(256),
                        nn.ReLU()),
                    merge_module=SumModule()
                    ),
                nn.AvgPool2d(8),
                nn.Flatten(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(256, 10)
                )
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path))
    return model

def le_net(pretrained = False, pretrained_path = None, **kwargs):
    """Constructs a LeNet model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on CIFAR10
    """
    channel_size = kwargs.get('channel_size',1)
    dropout = kwargs.get('dropout',0.25)
    num_classes = kwargs.get('num_classes',10)
    print('Building LeNet')
    print(f'Parameters {channel_size=}, {dropout=}, {num_classes=}')

    model = nn.Sequential(nn.Conv2d(channel_size, 32, 3, 1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, 1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Flatten(),
                    nn.Linear(12544, 128),
                    nn.Dropout(2* dropout) if dropout > 0 else nn.Identity(),
                    nn.Linear(128, num_classes))
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path))
    return model




def baidu_sequential(pretrained = False, pretrained_path = None, **kwargs):
    channel_size = kwargs.get('channel_size',3)
    dropout = kwargs.get('dropout',0.25)
    batchnorm = kwargs.get('batchnorm',True)
    num_classes = kwargs.get('num_classes',10)
    
    print(f'Building BaiduNet')
    print(f'Parameters {channel_size=}, {dropout=}, {batchnorm=}, {num_classes=}')
    
    model =  nn.Sequential(
                nn.Conv2d(channel_size, 64, kernel_size=3, stride=1, padding=1, bias=False),
                
                nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
                
                nn.ReLU(),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.MaxPool2d(2),
                Parallel(
                    nn.Identity(),
                    nn.Sequential(
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False),
                        nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
                        nn.ReLU(),
                        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False),
                        nn.BatchNorm2d(128) if batchnorm else nn.Identity(),
                        nn.ReLU()),
                    merge_module=SumModule()
                ),
                
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
                nn.ReLU(),
                
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.MaxPool2d(2),
                Parallel(
                    nn.Identity(),
                    nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
                        nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
                        nn.ReLU(),
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
                        nn.BatchNorm2d(256) if batchnorm else nn.Identity(),
                        nn.ReLU()),
                    merge_module=SumModule()
                    ),
                nn.AvgPool2d(8),
                nn.Flatten(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(256, num_classes)
                )
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path))
    return model




class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        DROPOUT = 0.1

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout2 = nn.Dropout(DROPOUT)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(DROPOUT)
            )

    def forward(self, x):
        out = F.relu(self.dropout1(self.bn1(self.conv1(x))))
        out = self.dropout2(self.bn2(self.conv2(out)))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,input_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
        
        self.bn1 = nn.BatchNorm2d(64)
        
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        
        self.pre = nn.Sequential(
            self.conv1,
            self.bn1
        )
        
        self.layers = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )
        
        
        
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(pretrained = False, pretrained_path = None, **kwargs):
    input_channels = kwargs.get('channel_size',3)
    num_classes = kwargs.get('num_classes',10)
    print('Building ResNet18')
    print(f'Parameters {input_channels=}')
    model =  ResNet(BasicBlock, [2, 2, 2, 2],input_channels=input_channels,num_classes=num_classes)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path))
    return model


def resnet_sequential(model: ResNet, pretrained = False, pretrained_path = None):
    model_seq = nn.Sequential(*list(model.pre), nn.ReLU(),*list(model.layers), nn.AvgPool2d(4), nn.Flatten(), model.linear)
    if pretrained:
        model_seq.load_state_dict(torch.load(pretrained_path))
    return model_seq

    
    


    
    

'''
Modified from https://github.com/pytorch/vision.git
'''


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
        


    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def make_layers(cfg, in_channels = 3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

def vgg11(**kwargs):
    batch_norm = kwargs.get('batchnorm', True)
    in_channels = kwargs.get('in_channels', 3)
    num_classes = kwargs.get('num_classes',10)
    """VGG 11-layer model (configuration "A")"""
    print(f'Building VGG11')
    print(f'Parameters: {batch_norm=} {in_channels=}')
    return VGG(make_layers(cfg['A'],batch_norm=batch_norm,in_channels=in_channels),num_classes=num_classes)


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))

def xavier_uniform_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model

def xavier_normal_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model

def kaiming_uniform_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu', a=0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model

def kaiming_normal_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu', a=0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model

def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")"""
    
    batch_norm = kwargs.get('batchnorm', True)
    in_channels = kwargs.get('in_channels', 3)
    num_classes = kwargs.get('num_classes',10)
    print(f'Building VGG16')
    print(f'Parameters: {batch_norm=} {in_channels=}')
    return VGG(make_layers(cfg['D'],batch_norm=batch_norm,in_channels=in_channels),num_classes=num_classes)


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")"""
    batch_norm = kwargs.get('batchnorm', True)
    in_channels = kwargs.get('in_channels', 3)
    
    return VGG(make_layers(cfg['E'],batch_norm=batch_norm,in_channels=in_channels))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))

def vgg_sequential(model: VGG, pretrained = False, pretrained_path = None):
    model_seq =  nn.Sequential(*list(model.features), nn.Flatten(), *list(model.classifier))
    if pretrained:
        model_seq.load_state_dict(torch.load(pretrained_path))
    return model_seq



def pruned_model(model: nn.Module, pruned_model: nn.Module, pruned_layers: list):
    '''
    Prune a model by removing layers in pruned_layers
    '''
    pruned_layers = set(pruned_layers)
    for name, module in model.named_children():
        if name in pruned_layers:
            continue
        pruned_model.add_module(name, module)
    return pruned_model



"""MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=planes,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=10,channel_size = 3):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(channel_size, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(
            320, 1280, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    
def MobileNetV2_sequential(model: MobileNetV2, pretrained = False, pretrained_path = None):
    model_seq = nn.Sequential(model.conv1,model.bn1, nn.ReLU(), *list(model.layers), model.conv2, model.bn2, nn.ReLU(), nn.AvgPool2d(4), nn.Flatten(), model.linear)
    if pretrained:
        model_seq.load_state_dict(torch.load(pretrained_path))
    return model_seq


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

import math


import torch
import torch.nn.functional as F
from torch import nn

class SGDRScheduler(nn.Module):
    global_epoch = 0
    all_epoch = 0
    cur_drop_prob = 0.
    def __init__(self, dropblock):
        super(SGDRScheduler, self).__init__()
        self.dropblock = dropblock
        self.drop_values = 0.

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        #self.dropblock.drop_prob = np.abs((0 + 0.5 * 0.1 * (1 + np.cos(np.pi * SGDRScheduler.global_epoch / SGDRScheduler.all_epoch)))-0.1)
        #SGDRScheduler.cur_drop_prob = self.dropblock.drop_prob
        ix = np.log2(self.global_epoch / 10 + 1).astype(int)
        T_cur = self.global_epoch - 10 * (2 ** (ix) - 1)
        T_i = (10 * 2 ** ix)
        self.dropblock.drop_prob = np.abs((0 + 0.5 * 0.1 * (1 + np.cos(np.pi * T_cur / T_i)))-0.1)
        SGDRScheduler.cur_drop_prob = self.dropblock.drop_prob

class LinearScheduler(nn.Module):
    global_epoch = 0
    num_epochs = 0
    def __init__(self, dropblock, start_value=0., stop_value=0.1):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=self.num_epochs)

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
            self.dropblock.drop_prob = self.drop_values[self.global_epoch]


class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class DropBlock3D(DropBlock2D):
    r"""Randomly zeroes 3D spatial blocks of the input tensor.
    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__(drop_prob, block_size)

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask[:, None, :, :, :],
                                  kernel_size=(self.block_size, self.block_size, self.block_size),
                                  stride=(1, 1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 3)

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, dilation=1):
        super(BasicConv, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation=dilation, groups=groups, bias=False)


    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class _SMG(nn.Module):
    def __init__(self, in_channels, growth_rate,
                 bn_size=4, groups=4, reduction_factor=2, forget_factor=2):
        super(_SMG, self).__init__()
        self.in_channels = in_channels
        self.reduction_factor = reduction_factor
        self.forget_factor = forget_factor
        self.growth_rate = growth_rate
        self.conv1_1x1 = BasicConv(in_channels, bn_size * growth_rate, kernel_size=1, stride=1)
        self.conv2_3x3 = BasicConv(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1,
                                   padding=1, groups=groups)

        # Mobile
        self.conv_3x3 = BasicConv(growth_rate, growth_rate, kernel_size=3,
                                  stride=1, padding=1, groups=growth_rate,)
        self.conv_5x5 = BasicConv(growth_rate, growth_rate, kernel_size=3,
                                  stride=1, padding=2, groups=growth_rate, dilation=2)

        # GTSK layers
        self.global_context3x3 = nn.Conv2d(growth_rate, 1, kernel_size=1)
        self.global_context5x5 = nn.Conv2d(growth_rate, 1, kernel_size=1)

        self.fcall = nn.Conv2d(2 * growth_rate, 2 * growth_rate // self.reduction_factor, kernel_size=1)
        self.bn_attention = nn.BatchNorm1d(2 * growth_rate // self.reduction_factor)
        self.fc3x3 = nn.Conv2d(2 * growth_rate // self.reduction_factor, growth_rate, kernel_size=1)
        self.fc5x5 = nn.Conv2d(2 * growth_rate // self.reduction_factor, growth_rate, kernel_size=1)

        # SE layers
        self.global_forget_context = nn.Conv2d(growth_rate, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_forget = nn.BatchNorm1d(growth_rate // self.forget_factor)
        self.fc1 = nn.Conv2d(growth_rate, growth_rate // self.forget_factor, kernel_size=1)
        self.fc2 = nn.Conv2d(growth_rate // self.forget_factor, growth_rate, kernel_size=1)

    def forward(self, x):
        x_dense = x
        x = self.conv1_1x1(x)
        x = self.conv2_3x3(x)

        H = W = x.size(-1)
        C = x.size(1)
        x_shortcut = x

        forget_context_weight = self.global_forget_context(x_shortcut)
        forget_context_weight = torch.flatten(forget_context_weight, start_dim=1)
        forget_context_weight = F.softmax(forget_context_weight, 1).reshape(-1, 1, H, W)
        x_shortcut_weight = self.global_pool(x_shortcut * forget_context_weight) * H * W

        x_shortcut_weight = \
            torch.tanh(self.bn_forget(torch.flatten(self.fc1(x_shortcut_weight), start_dim=1))) \
                .reshape(-1, C // self.forget_factor, 1, 1)
        x_shortcut_weight = torch.sigmoid(self.fc2(x_shortcut_weight))


        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        context_weight_3x3 = \
            F.softmax(torch.flatten(self.global_context3x3(x_3x3), start_dim=1), 1).reshape(-1, 1, H, W)
        context_weight_5x5 = \
            F.softmax(torch.flatten(self.global_context5x5(x_5x5), start_dim=1), 1).reshape(-1, 1, H, W)
        x_3x3 = self.global_pool(x_3x3 * context_weight_3x3) * H * W
        x_5x5 = self.global_pool(x_5x5 * context_weight_5x5) * H * W
        x_concat = torch.cat([x_3x3, x_5x5], 1)
        attention = torch.tanh(self.bn_attention(torch.flatten(self.fcall(x_concat), start_dim=1))) \
            .reshape(-1, 2 * C // self.reduction_factor, 1, 1)
        weight_3x3 = torch.unsqueeze(torch.flatten(self.fc3x3(attention), start_dim=1), 1)
        weight_5x5 = torch.unsqueeze(torch.flatten(self.fc5x5(attention), start_dim=1), 1)
        weight_all = F.softmax(torch.cat([weight_3x3, weight_5x5], 1), 1)
        weight_3x3, weight_5x5 = weight_all[:, 0, :].reshape(-1, C, 1, 1), weight_all[:, 1, :].reshape(-1, C, 1, 1)
        new_x = weight_3x3 * x_3x3 + weight_5x5 * x_5x5
        x = x_shortcut * x_shortcut_weight + new_x

        return torch.cat([x_dense, x], 1)


class _HybridBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_HybridBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('SMG%d' % (i+1),
                            _SMG(in_channels+growth_rate*i,
                                        growth_rate, bn_size))


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels, forget_factor=4, reduction_factor=4):
        super(_Transition, self).__init__()
        self.in_channels = in_channels
        self.forget_factor = forget_factor
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.reduce_channels = (in_channels - out_channels) // 2
        self.conv1_1x1 = BasicConv(in_channels, in_channels-self.reduce_channels, kernel_size=1, stride=1)
        self.conv2_3x3 = BasicConv(in_channels-self.reduce_channels, out_channels, kernel_size=3, stride=2,
                                   padding=1, groups=1)
        # Mobile
        # Mobile
        self.conv_3x3 = BasicConv(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, groups=out_channels)
        self.conv_5x5 = BasicConv(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=2, dilation=2, groups=out_channels)

        # GTSK layers
        self.global_context3x3 = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.global_context5x5 = nn.Conv2d(out_channels, 1, kernel_size=1)

        self.fcall = nn.Conv2d(2 * out_channels, 2 * out_channels // self.reduction_factor, kernel_size=1)
        self.bn_attention = nn.BatchNorm1d(2 * out_channels // self.reduction_factor)
        self.fc3x3 = nn.Conv2d(2 * out_channels // self.reduction_factor, out_channels, kernel_size=1)
        self.fc5x5 = nn.Conv2d(2 * out_channels // self.reduction_factor, out_channels, kernel_size=1)

        # SE layers
        self.global_forget_context = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_forget = nn.BatchNorm1d(out_channels // self.forget_factor)
        self.fc1 = nn.Conv2d(out_channels, out_channels // self.forget_factor, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channels // self.forget_factor, out_channels, kernel_size=1)
        self.dropblock = SGDRScheduler(DropBlock2D(drop_prob=0, block_size=2))


    def forward(self, x):
        self.dropblock.step()
        x = self.conv1_1x1(x)
        x = self.conv2_3x3(x)

        H = W = x.size(-1)
        C = x.size(1)
        x_shortcut = x

        forget_context_weight = self.global_forget_context(x_shortcut)
        forget_context_weight = torch.flatten(forget_context_weight, start_dim=1)
        forget_context_weight = F.softmax(forget_context_weight, 1)
        forget_context_weight = forget_context_weight.reshape(-1, 1, H, W)
        x_shortcut_weight = self.global_pool(x_shortcut * forget_context_weight) * H * W

        x_shortcut_weight = \
            torch.tanh(self.bn_forget(torch.flatten(self.fc1(x_shortcut_weight), start_dim=1))) \
                .reshape(-1, C // self.forget_factor, 1, 1)
        x_shortcut_weight = torch.sigmoid(self.fc2(x_shortcut_weight))


        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        context_weight_3x3 = \
            F.softmax(torch.flatten(self.global_context3x3(x_3x3), start_dim=1), 1).reshape(-1, 1, H, W)
        context_weight_5x5 = \
            F.softmax(torch.flatten(self.global_context5x5(x_5x5), start_dim=1), 1).reshape(-1, 1, H, W)
        x_3x3 = self.global_pool(x_3x3 * context_weight_3x3) * H * W
        x_5x5 = self.global_pool(x_5x5 * context_weight_5x5) * H * W
        x_concat = torch.cat([x_3x3, x_5x5], 1)
        attention = torch.tanh(self.bn_attention(torch.flatten(self.fcall(x_concat), start_dim=1))) \
            .reshape(-1, 2 * C // self.reduction_factor, 1, 1)
        weight_3x3 = torch.unsqueeze(torch.flatten(self.fc3x3(attention), start_dim=1), 1)
        weight_5x5 = torch.unsqueeze(torch.flatten(self.fc5x5(attention), start_dim=1), 1)
        weight_all = F.softmax(torch.cat([weight_3x3, weight_5x5], 1), 1)
        weight_3x3, weight_5x5 = weight_all[:, 0, :].reshape(-1, C, 1, 1), weight_all[:, 1, :].reshape(-1, C, 1, 1)
        new_x = weight_3x3 * x_3x3 + weight_5x5 * x_5x5

        x = x_shortcut * x_shortcut_weight + new_x

        return self.dropblock(x)

        #return x

class HCGNet(nn.Module):
    def __init__(self, growth_rate=(8, 16, 32), block_config=(6,12,24,16),
                 bn_size=4, theta=0.5, num_classes=10):
        super(HCGNet, self).__init__()
        num_init_feature = 2 * growth_rate[0]

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_feature,
                                kernel_size=3, stride=1,
                                padding=1, bias=False)),
        ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('HybridBlock%d' % (i+1),
                                     _HybridBlock(num_layers, num_feature, bn_size, growth_rate[i]))
            num_feature = num_feature + growth_rate[i] * num_layers
            if i != len(block_config)-1:
                self.features.add_module('Transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.classifier = nn.Linear(num_feature, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = F.adaptive_avg_pool2d(F.relu(features),(1, 1))
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


def HCGNet_A1(num_classes=100):
    return HCGNet(growth_rate=(12, 24, 36), block_config=(8, 8, 8), num_classes=num_classes)


def HCGNet_A2(num_classes=100):
    return HCGNet(growth_rate=(24, 36, 64), block_config=(8, 8, 8), num_classes=num_classes)


def HCGNet_A3(num_classes=100):
    return HCGNet(growth_rate=(36, 48, 80), block_config=(12, 12, 12),num_classes=num_classes)


def HCGNet_sequential(model: HCGNet, pretrained = False, pretrained_path = None):
    model_seq = nn.Sequential(model.module.features,nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), model.module.classifier)
    if pretrained:
        model_seq.load_state_dict(torch.load(pretrained_path))
    return model_seq


### {\url{https://github.com/chenyaofo/pytorch-cifar-models }

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class CifarBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CifarBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CifarResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def cifar_resnet56(num_classes=100):
    return CifarResNet(CifarBasicBlock, [9, 9, 9], num_classes=num_classes)

def CifarResNet_sequential(model: CifarResNet):
    model_seq = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.layer1,
        model.layer2,
        model.layer3,
        model.avgpool,
        nn.Flatten(),
        model.fc
    )
    return model_seq



# --- CifarMobileNetV2 (chenyaofo/pytorch-cifar-models) ---
# Versione di MobileNetV2 adattata per CIFAR (stride 1 invece di 2 nel primo layer)
# Diversa dalla MobileNetV2 già presente in nicolo_net.py (che usa Block custom)
# Aggiunta per supportare i pesi pretrained di chenyaofo su CIFAR100

class CifarVGG(nn.Module):
    def __init__(self, features, num_classes=100):
        super(CifarVGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def cifar_vgg13_bn(num_classes=100):
    return CifarVGG(make_layers(cfg['B'], batch_norm=True), num_classes=num_classes)



def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 groups=1, norm_layer=None, activation_layer=None, dilation=1):
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      dilation=dilation, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes

ConvBNReLU = ConvBNActivation

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class CifarMobileNetV2(nn.Module):
    def __init__(self, num_classes=100, width_mult=1.0, norm_layer=None):
        super(CifarMobileNetV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        features = [ConvBNReLU(3, input_channel, stride=1, norm_layer=norm_layer)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def cifar_mobilenetv2_x1_0(num_classes=100):
    return CifarMobileNetV2(num_classes=num_classes, width_mult=1.0)

def cifar_mobilenetv2_x1_4(num_classes=100):
    return CifarMobileNetV2(num_classes=num_classes, width_mult=1.4)

def CifarMobileNetV2_sequential(model: CifarMobileNetV2):
    model_seq = nn.Sequential(
        model.features,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        model.classifier[1]  # solo il Linear, skippa il Dropout
    )
    return model_seq

# --- CifarShuffleNetV2 (chenyaofo/pytorch-cifar-models) ---
# Versione di ShuffleNetV2 adattata per CIFAR (stride 1 invece di 2 nel conv1, maxpool rimosso)
# Aggiunta per supportare i pesi pretrained di chenyaofo su CIFAR100

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    # Rimescola i canali tra i gruppi per favorire lo scambio di informazioni
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class CifarInvertedResidual(nn.Module):
    # Blocco residuale con due branch: branch1 (downsampling) e branch2 (trasformazione)
    # Se stride=1 splitta i canali a metà, se stride=2 usa entrambi i branch
    def __init__(self, inp, oup, stride):
        super(CifarInvertedResidual, self).__init__()
        self.stride = stride
        branch_features = oup // 2
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride=self.stride, padding=1, bias=False, groups=inp),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride=self.stride, padding=1, bias=False, groups=branch_features),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class CifarShuffleNetV2(nn.Module):
    # Architettura completa: conv1 + stage2/3/4 + conv5 + fc
    # network[-1] = fc (Linear finale), compatibile con ABP
    def __init__(self, stages_repeats, stages_out_channels, num_classes=100):
        super(CifarShuffleNetV2, self).__init__()
        self._stage_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        stage_names = ['stage2', 'stage3', 'stage4']
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [CifarInvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(CifarInvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global average pooling
        x = self.fc(x)
        return x


def cifar_shufflenetv2_x2_0(num_classes=100):
    return CifarShuffleNetV2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 244, 488, 976, 2048],
        num_classes=num_classes
    )

def cifar_shufflenetv2_x1_0(num_classes=100):
    return CifarShuffleNetV2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 116, 232, 464, 1024],
        num_classes=num_classes
    )


def CifarShuffleNetV2_sequential(model: CifarShuffleNetV2):
    # Wrappa il modello in nn.Sequential per compatibilità con ABP
    # network[-1] = fc (Linear finale)
    model_seq = nn.Sequential(
        model.conv1,
        model.stage2,
        model.stage3,
        model.stage4,
        model.conv5,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        model.fc
    )
    return model_seq