### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from typing import Union
import pathlib

### External Imports ###
import numpy as np
import pandas as pd
import torch as tc
import torchio as tio
from torch.utils.tensorboard import SummaryWriter

### MONAI Imports ###
from monai import transforms as mtr

### Internal Imports ###
from paths import pc_paths as p
from helpers import cost_functions as cf
from datasets import dataset as ds
from networks import runet, segmamba
########################




def exp_1(fold=1):
    """
    """
    ### Dataset Params ###
    training_dataset_path = p.parsed_data_path / "Spacing_05_05_05"
    validation_dataset_path = p.parsed_data_path / "Spacing_05_05_05"
    training_csv_path = p.csv_path / f"training_fold_{fold}.csv"
    validation_csv_path = p.csv_path / f"val_fold_{fold}.csv"
    mode = 'pre'

    ### Prepare Data ###
    training_data = ds.prepare_data(training_dataset_path, training_csv_path, mode)
    validation_data = ds.prepare_data(validation_dataset_path, validation_csv_path, mode)

    ### Prepare Loading & Augmentation ###
    samples_per_volume = 4
    batch_size = 1
    patch_size = (128, 128, 128)
    num_workers = 8
    training_cache_rate = 0.05
    validation_cache_rate = 0.05
    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.Orientationd(keys=["image", "gt"], axcodes="RAS"),
        mtr.NormalizeIntensityd(keys=["image"]),
        mtr.RandCropByPosNegLabeld(
            keys=["image", "gt"],
            label_key="gt",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=samples_per_volume
        )
    ])

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.Orientationd(keys=["image", "gt"], axcodes="RAS"),
        mtr.NormalizeIntensityd(keys=["image"])
    ])

    ### General Parameters ###
    experiment_name = f"HNTSMRG_PRE_Exp_1_F{fold}"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': False}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0, 1]
    logger = None
    callbacks = None
    max_epochs = 201
    precision = "bf16-mixed"
    strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = runet.default_pre_config()
    config['use_sigmoid'] = False
    model = runet.RUNet(**config)

    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['precision'] = precision
    lighting_params['strategy'] = strategy
    lighting_params['deterministic'] = deterministic

    ### Parse Parameters ###
    training_params = dict()
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['model'] = model
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['echo'] = echo
    training_params['num_iterations'] = max_epochs
    training_params['lightning_params'] = lighting_params
    training_params['patch_size'] = patch_size

    training_params['training_data'] = training_data
    training_params['validation_data'] = validation_data
    training_params['training_transforms'] = training_transforms
    training_params['validation_transforms'] = validation_transforms
    training_params['training_cache_rate'] = training_cache_rate
    training_params['validation_cache_rate'] = validation_cache_rate
    training_params['num_workers'] = num_workers
    training_params['batch_size'] = batch_size
    training_params['num_workers'] = num_workers
    training_params['batch_size'] = batch_size
    ### Cost functions and params
    training_params['objective_function'] = objective_function
    training_params['objective_function_params'] = objective_function_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['lightning_params'] = lighting_params

    ########################################
    return training_params




def exp_2(fold=1):
    """
    """
    ### Dataset Params ###
    training_dataset_path = p.parsed_data_path / "Spacing_05_05_05"
    validation_dataset_path = p.parsed_data_path / "Spacing_05_05_05"
    training_csv_path = p.csv_path / f"training_fold_{fold}.csv"
    validation_csv_path = p.csv_path / f"val_fold_{fold}.csv"
    mode = 'pre'

    ### Prepare Data ###
    training_data = ds.prepare_data(training_dataset_path, training_csv_path, mode)
    validation_data = ds.prepare_data(validation_dataset_path, validation_csv_path, mode)

    ### Prepare Loading & Augmentation ###
    samples_per_volume = 4
    batch_size = 1
    patch_size = (128, 128, 128)
    num_workers = 8
    training_cache_rate = 0.05
    validation_cache_rate = 0.05
    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.Orientationd(keys=["image", "gt"], axcodes="RAS"),
        mtr.NormalizeIntensityd(keys=["image"]),
        mtr.RandCropByPosNegLabeld(
            keys=["image", "gt"],
            label_key="gt",
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=samples_per_volume
        )
    ])

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.Orientationd(keys=["image", "gt"], axcodes="RAS"),
        mtr.NormalizeIntensityd(keys=["image"])
    ])

    ### General Parameters ###
    experiment_name = f"HNTSMRG_PRE_Exp_2_F{fold}"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': False}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0, 1]
    logger = None
    callbacks = None
    max_epochs = 501
    precision = "bf16-mixed"
    strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = segmamba.default_pre_config()
    model = segmamba.SegMamba(**config)

    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['precision'] = precision
    lighting_params['strategy'] = strategy
    lighting_params['deterministic'] = deterministic

    ### Parse Parameters ###
    training_params = dict()
    ### General params
    training_params['experiment_name'] = experiment_name
    training_params['model'] = model
    training_params['learning_rate'] = learning_rate
    training_params['to_load_checkpoint_path'] = to_load_checkpoint_path
    training_params['save_step'] = save_step
    training_params['number_of_images_to_log'] = number_of_images_to_log
    training_params['echo'] = echo
    training_params['num_iterations'] = max_epochs
    training_params['lightning_params'] = lighting_params
    training_params['patch_size'] = patch_size

    training_params['training_data'] = training_data
    training_params['validation_data'] = validation_data
    training_params['training_transforms'] = training_transforms
    training_params['validation_transforms'] = validation_transforms
    training_params['training_cache_rate'] = training_cache_rate
    training_params['validation_cache_rate'] = validation_cache_rate
    training_params['num_workers'] = num_workers
    training_params['batch_size'] = batch_size
    training_params['num_workers'] = num_workers
    training_params['batch_size'] = batch_size
    ### Cost functions and params
    training_params['objective_function'] = objective_function
    training_params['objective_function_params'] = objective_function_params
    training_params['optimizer_weight_decay'] = optimizer_weight_decay
    training_params['lightning_params'] = lighting_params

    ########################################
    return training_params