### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Iterable, Union, Sequence, Tuple
import math
import itertools

### External Imports ###
import torch as tc
import numpy as np
import torch.nn.functional as F
import torchsummary as ts
import einops

from monai.networks.nets import unetr

### Internal Imports ###


########################

def default_pre_config():
    config = {}
    config['img_size'] = (128, 128, 128)
    config['spatial_dims'] = 3
    config['in_channels'] = 1
    config['out_channels'] = 3
    config['feature_size'] = 16
    config['hidden_size'] = 768
    config['mlp_dim'] = 3072
    config['num_heads'] = 12
    return config

def default_mid_config():
    config = {}
    config['img_size'] = (128, 128, 128)
    config['spatial_dims'] = 3
    config['in_channels'] = 3
    config['out_channels'] = 3
    config['feature_size'] = 16
    config['hidden_size'] = 768
    config['mlp_dim'] = 3072
    config['num_heads'] = 12
    return config

class UNETR(tc.nn.Module):
    def __init__(self, **config):
        super().__init__()   
        self.image_size = config['img_size']
        self.model = unetr.UNETR(**config)
    
    def forward(self, x : tc.Tensor) -> tc.Tensor:
        _, _, d, h, w = x.shape
        if self.image_size is not None and (d, h, w) != (self.image_size[0], self.image_size[1], self.image_size[2]):
            x = F.interpolate(x, self.image_size, mode='trilinear')
        x = self.model(x.contiguous())
        x = tc.nn.Sigmoid()(x)
        if self.image_size is not None and (d, h, w) != (self.image_size[0], self.image_size[1], self.image_size[2]):
            x = F.interpolate(x, (d, h, w), mode='trilinear')
        return x
             
def create_model(**config) -> tc.nn.Module:
    return UNETR(**config)


def test_1():
    device = "cuda:0"
    config = default_pre_config()
    model = create_model(**config)
    model = model.to(device)
    num_samples = 1
    num_channels = 1
    y_size = 128
    x_size = 128
    z_size = 128
    input = tc.randn((num_samples, num_channels, y_size, x_size, z_size), device=device)
    result = model(input.contiguous())
    print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=[input], device=device, depth=10)  
    
def test_2():
    device = "cuda:0"
    config = default_mid_config()
    model = create_model(**config)
    model = model.to(device)
    num_samples = 1
    num_channels = 3
    y_size = 128
    x_size = 128
    z_size = 128
    input = tc.randn((num_samples, num_channels, y_size, x_size, z_size), device=device)
    result = model(input.contiguous())
    print(f"Result shape: {result.shape}")
    ts.summary(model, input_data=[input], device=device, depth=10)    




def run():
    test_1()
    test_2()
    pass

if __name__ == "__main__":
    run()