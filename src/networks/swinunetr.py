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

from monai.networks.nets import swin_unetr

### Internal Imports ###


########################


def default_pre_config():
    config = {}
    config['img_size'] = (128, 128, 128)
    config['in_channels'] = 1
    config['out_channels'] = 3
    config['depths'] = (2, 2, 2, 2)
    config['num_heads'] = (3, 6, 12, 24)
    config['feature_size'] = 48
    config['norm_name'] = "instance"
    config['drop_rate'] = 0.0
    config['attn_drop_rate'] = 0.0
    config['dropout_path_rate'] = 0.0
    config['normalize'] = True
    config['use_checkpoint'] = False
    config['spatial_dims'] = 3
    config['downsample'] = "mergingv2"
    return config

def default_mid_config():
    config = {}
    config['img_size'] = (128, 128, 128)
    config['in_channels'] = 3
    config['out_channels'] = 3
    config['depths'] = (2, 2, 2, 2)
    config['num_heads'] = (3, 6, 12, 24)
    config['feature_size'] = 48
    config['norm_name'] = "instance"
    config['drop_rate'] = 0.0
    config['attn_drop_rate'] = 0.0
    config['dropout_path_rate'] = 0.0
    config['normalize'] = True
    config['use_checkpoint'] = False
    config['spatial_dims'] = 3
    config['downsample'] = "mergingv2"
    return config

class SwinUNetR(tc.nn.Module):
    def __init__(self, **config):
        super().__init__()   
        self.image_size = config['img_size']
        self.model = swin_unetr.SwinUNETR(**config)
    
    def forward(self, x : tc.Tensor) -> tc.Tensor:
        _, _, d, h, w = x.shape
        if self.image_size is not None and (d, h, w) != (self.image_size[0], self.image_size[1], self.image_size[2]):
            x = F.interpolate(x, self.image_size, mode='trilinear')
        x = self.model(x.contiguous())
        if self.image_size is not None and (d, h, w) != (self.image_size[0], self.image_size[1], self.image_size[2]):
            x = F.interpolate(x, (d, h, w), mode='trilinear')
        return x
             
def create_model(**config) -> tc.nn.Module:
    return SwinUNetR(**config)

def test_1():
    device = "cuda:0"
    config = default_pre_config()
    model = create_model(**config)
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
    # test_1()
    test_2()
    pass

if __name__ == "__main__":
    run()