### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from typing import Union
import pathlib
import psutil
import time

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


### MONAI Imports ###
from monai import transforms as mtr

### Internal Imports ###
from paths import hpc_paths as p
from helpers import cost_functions as cf
from networks import runet
from networks import swinunetr, unetr, attentionunet
from datasets import dataset as ds

########################

def load_checkpoint(model, checkpoint_path, mode):
    if mode == "pre":
        state_dict = tc.load(checkpoint_path, map_location="cpu", weights_only=False)['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict[key.replace("model.", "", 1)] = state_dict[key]
        model.load_state_dict(new_state_dict)
    elif mode == "mid":
        state_dict = tc.load(checkpoint_path, map_location="cpu", weights_only=False)['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict[key.replace("model.", "", 1)] = state_dict[key]
        model.load_state_dict(new_state_dict)
    return model



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
    samples_per_volume = 8
    batch_size = 1
    patch_size = (128, 128, 128)
    num_workers = 16
    training_cache_rate = 0.25
    validation_cache_rate = 1.0
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
    experiment_name = f"H_PRE_Exp1_F{fold}"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': False}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0, 1, 2, 3]
    logger = None
    callbacks = None
    max_epochs = 501
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




def exp_1a(fold=1):
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
    training_cache_rate = 0.25
    validation_cache_rate = 1.0
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
    experiment_name = f"H_PRE_Exp1a_F{fold}"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': False}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0, 1, 2, 3]
    logger = None
    callbacks = None
    max_epochs = 501
    # precision = "bf16-mixed"
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
    # lighting_params['precision'] = precision
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



def exp_1b(fold=1):
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
    num_workers = 1
    training_cache_rate = 0.25
    validation_cache_rate = 1.0
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
    experiment_name = f"H_PRE_Exp1b_F{fold}"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': False}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0, 1, 2, 3]
    logger = None
    callbacks = None
    max_epochs = 501
    # precision = "bf16-mixed"
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
    # lighting_params['precision'] = precision
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






def exp_1c(fold=1):
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
    samples_per_volume = 8
    batch_size = 1
    patch_size = (128, 128, 128)
    num_workers = 16
    training_cache_rate = 0.25
    validation_cache_rate = 1.0
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
    experiment_name = f"H_PRE_Exp1c_F{fold}"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': False}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0, 1, 2, 3]
    logger = None
    callbacks = None
    max_epochs = 501
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









def exp_1d(fold=1):
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
    num_workers = 16
    training_cache_rate = 1.0
    validation_cache_rate = 1.0
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
    experiment_name = f"H_PRE_Exp1d_F{fold}"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': False}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0]
    logger = None
    callbacks = None
    max_epochs = 501
    # precision = "bf16-mixed"
    # strategy = "ddp"
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
    # lighting_params['precision'] = precision
    # lighting_params['strategy'] = strategy
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




def exp_1e(fold=1):
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
    num_workers = 16
    training_cache_rate = 1.0
    validation_cache_rate = 1.0
    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        # mtr.Orientationd(keys=["image", "gt"], axcodes="RAS"),
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
        # mtr.Orientationd(keys=["image", "gt"], axcodes="RAS"),
        mtr.NormalizeIntensityd(keys=["image"])
    ])

    ### General Parameters ###
    experiment_name = f"H_PRE_Exp1e_F{fold}_2"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': True}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0]
    logger = None
    callbacks = None
    max_epochs = 501
    # precision = "bf16-mixed"
    # strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = runet.default_pre_config()
    config['use_sigmoid'] = False
    model = runet.RUNet(**config)
    checkpoint_path = p.checkpoints_path / f"H_PRE_Exp1e_F{fold}" / 'epoch=498_dice.ckpt'
    model = load_checkpoint(model, checkpoint_path, mode)

    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    # lighting_params['precision'] = precision
    # lighting_params['strategy'] = strategy
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




def exp_1f(fold=1):
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
    num_workers = 16
    training_cache_rate = 1.0
    validation_cache_rate = 1.0

    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        # mtr.Orientationd(keys=["image", "gt"], axcodes="RAS"),
        mtr.NormalizeIntensityd(keys=["image"]),
        mtr.RandAxisFlipd(keys=["image", "gt"], prob=0.5),
        mtr.RandZoomd(keys=["image", "gt"], prob=0.5, min_zoom=0.7, max_zoom=1.3, mode=['area', 'nearest'], padding_mode=['constant', 'constant']),
        mtr.RandGaussianNoised(keys=["image"], prob=0.5, mean=0, std=0.1),
        mtr.RandRicianNoised(keys=["image"], prob=0.2, mean=0, std=0.2),
        mtr.RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
        mtr.RandCropByPosNegLabeld(
            keys=["image", "gt"],
            label_key="gt",
            spatial_size=patch_size,
            pos=1,
            neg=2,
            num_samples=samples_per_volume
        )
    ])

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        # mtr.Orientationd(keys=["image", "gt"], axcodes="RAS"),
        mtr.NormalizeIntensityd(keys=["image"])
    ])

    ### General Parameters ###
    experiment_name = f"H_PRE_Exp1f_F{fold}"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': False}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0]
    logger = None
    callbacks = None
    max_epochs = 501
    # precision = "bf16-mixed"
    # strategy = "ddp"
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
    # lighting_params['precision'] = precision
    # lighting_params['strategy'] = strategy
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










def exp_1g(fold=1):
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
    num_workers = 16
    training_cache_rate = 1.0
    validation_cache_rate = 1.0

    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.NormalizeIntensityd(keys=["image"]),
        mtr.RandCropByLabelClassesd(
            keys=["image", "gt"],
            label_key="gt",
            spatial_size=patch_size,
            ratios=[1, 1, 1],
            num_classes=3,
            num_samples=samples_per_volume)
    ])

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.NormalizeIntensityd(keys=["image"])
    ])

    ### General Parameters ###
    experiment_name = f"H_PRE_Exp1g_F{fold}_V2"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': True}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0]
    logger = None
    callbacks = None
    max_epochs = 501
    # precision = "bf16-mixed"
    # strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = runet.default_pre_config()
    config['use_sigmoid'] = False
    model = runet.RUNet(**config)
    checkpoints = {1: 'epoch=484_dice.ckpt', 2: 'epoch=388_dice.ckpt', 3: 'epoch=366_dice.ckpt', 4: 'epoch=464_dice.ckpt', 5: 'epoch=390_dice.ckpt'}
    checkpoint_path = p.checkpoints_path / f"H_PRE_Exp1g_F{fold}" / checkpoints[int(fold)]
    model = load_checkpoint(model, checkpoint_path, mode)

    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    # lighting_params['precision'] = precision
    # lighting_params['strategy'] = strategy
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








def exp_1h(fold=1):
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
    num_workers = 16
    training_cache_rate = 1.0
    validation_cache_rate = 1.0


    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.NormalizeIntensityd(keys=["image"]),
        mtr.RandAxisFlipd(keys=["image", "gt"], prob=0.5),
        mtr.RandGaussianNoised(keys=["image"], prob=0.5, mean=0, std=0.1),
        mtr.RandRicianNoised(keys=["image"], prob=0.2, mean=0, std=0.2),
        mtr.RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
        mtr.RandCropByLabelClassesd(
            keys=["image", "gt"],
            label_key="gt",
            spatial_size=patch_size,
            ratios=[1, 1, 1],
            num_classes=3,
            num_samples=samples_per_volume)
    ])

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.NormalizeIntensityd(keys=["image"])
    ])

    ### General Parameters ###
    experiment_name = f"H_PRE_Exp1h_F{fold}_V3"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': True}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0]
    logger = None
    callbacks = None
    max_epochs = 501
    # precision = "bf16-mixed"
    # strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = runet.default_pre_config()
    config['use_sigmoid'] = False
    model = runet.RUNet(**config)
    # checkpoints = {1: 'epoch=484_dice.ckpt', 2: 'epoch=388_dice.ckpt', 3: 'epoch=366_dice.ckpt', 4: 'epoch=464_dice.ckpt', 5: 'epoch=390_dice.ckpt'}
    # checkpoint_path = p.checkpoints_path / f"H_PRE_Exp1g_F{fold}" / checkpoints[int(fold)]
    # model = load_checkpoint(model, checkpoint_path, mode)

    # checkpoints = {1: 'epoch=174_dice.ckpt', 2: 'epoch=173_dice.ckpt', 3: 'epoch=169_dice.ckpt', 4: 'epoch=173_dice.ckpt', 5: 'epoch=178_dice.ckpt'}
    checkpoints = {1: 'epoch=182_dice.ckpt', 2: 'epoch=151_dice.ckpt', 3: 'epoch=181_dice.ckpt', 4: 'epoch=166_dice.ckpt', 5: 'epoch=171_dice.ckpt'}
    checkpoint_path = p.checkpoints_path / f"H_PRE_Exp1h_F{fold}_V2" / checkpoints[int(fold)]
    model = load_checkpoint(model, checkpoint_path, mode)

    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    # lighting_params['precision'] = precision
    # lighting_params['strategy'] = strategy
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




















































def exp_2a(fold=1):
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
    num_workers = 16
    training_cache_rate = 1.0
    validation_cache_rate = 1.0

    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.NormalizeIntensityd(keys=["image"]),
        mtr.RandCropByLabelClassesd(
            keys=["image", "gt"],
            label_key="gt",
            spatial_size=patch_size,
            ratios=[1, 1, 1],
            num_classes=3,
            num_samples=samples_per_volume)
    ])

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.NormalizeIntensityd(keys=["image"])
    ])

    ### General Parameters ###
    experiment_name = f"H_PRE_Exp2a_F{fold}_V2"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': True}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0]
    logger = None
    callbacks = None
    max_epochs = 501
    # precision = "bf16-mixed"
    # strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = unetr.default_pre_config()
    model = unetr.UNETR(**config)
    checkpoints = {1: 'epoch=379_dice.ckpt'}
    checkpoint_path = p.checkpoints_path /  f"H_PRE_Exp2a_F{fold}" / checkpoints[int(fold)]
    model = load_checkpoint(model, checkpoint_path, mode)

    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    # lighting_params['precision'] = precision
    # lighting_params['strategy'] = strategy
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






def exp_2b(fold=1):
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
    num_workers = 16
    training_cache_rate = 1.0
    validation_cache_rate = 1.0

    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.NormalizeIntensityd(keys=["image"]),
        mtr.RandCropByLabelClassesd(
            keys=["image", "gt"],
            label_key="gt",
            spatial_size=patch_size,
            ratios=[1, 1, 1],
            num_classes=3,
            num_samples=samples_per_volume)
    ])

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.NormalizeIntensityd(keys=["image"])
    ])

    ### General Parameters ###
    experiment_name = f"H_PRE_Exp2b_F{fold}_V2"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': True}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0]
    logger = None
    callbacks = None
    max_epochs = 501
    # precision = "bf16-mixed"
    # strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = attentionunet.default_pre_config()
    model = attentionunet.AttentionUNet(**config)
    checkpoints = {1: 'epoch=190_dice.ckpt'}
    checkpoint_path = p.checkpoints_path /  f"H_PRE_Exp2b_F{fold}" / checkpoints[int(fold)]
    model = load_checkpoint(model, checkpoint_path, mode)

    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    # lighting_params['precision'] = precision
    # lighting_params['strategy'] = strategy
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



def exp_2c(fold=1):
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
    num_workers = 16
    training_cache_rate = 1.0
    validation_cache_rate = 1.0

    if int(fold) == 1:
        training_transforms = mtr.Compose([
            mtr.LoadImaged(keys=['image', 'gt']),
            mtr.EnsureChannelFirstd(keys=["image", "gt"]),
            mtr.NormalizeIntensityd(keys=["image"]),
            mtr.RandAxisFlipd(keys=["image", "gt"], prob=0.5),
            mtr.RandGaussianNoised(keys=["image"], prob=0.5, mean=0, std=0.1),
            mtr.RandRicianNoised(keys=["image"], prob=0.2, mean=0, std=0.2),
            mtr.RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
            mtr.RandCropByLabelClassesd(
                keys=["image", "gt"],
                label_key="gt",
                spatial_size=patch_size,
                ratios=[1, 1, 1],
                num_classes=3,
                num_samples=samples_per_volume)
        ])
    else:
        training_transforms = mtr.Compose([
            mtr.LoadImaged(keys=['image', 'gt']),
            mtr.EnsureChannelFirstd(keys=["image", "gt"]),
            mtr.NormalizeIntensityd(keys=["image"]),
            mtr.RandCropByLabelClassesd(
                keys=["image", "gt"],
                label_key="gt",
                spatial_size=patch_size,
                ratios=[1, 1, 1],
                num_classes=3,
                num_samples=samples_per_volume)
        ])    

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.NormalizeIntensityd(keys=["image"])
    ])

    ### General Parameters ###
    if int(fold) == 1:
        experiment_name = f"H_PRE_Exp2c_F{fold}_V3"
    else:
        experiment_name = f"H_PRE_Exp2c_F{fold}"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': True}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0]
    logger = None
    callbacks = None
    max_epochs = 501
    # precision = "bf16-mixed"
    # strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = swinunetr.default_pre_config()
    model = swinunetr.SwinUNetR(**config)
    # checkpoints = {1: 'epoch=343_dice.ckpt'}
    checkpoints = {1: 'epoch=477_dice.ckpt'}
    if int(fold) == 1:
        checkpoint_path = p.checkpoints_path /  f"H_PRE_Exp2c_F{fold}_V2" / checkpoints[int(fold)]
        model = load_checkpoint(model, checkpoint_path, mode)

    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    # lighting_params['precision'] = precision
    # lighting_params['strategy'] = strategy
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


























def exp_1h_multi(fold=1):
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
    num_workers = 16
    training_cache_rate = 0.25
    validation_cache_rate = 1.0


    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.NormalizeIntensityd(keys=["image"]),
        mtr.RandAxisFlipd(keys=["image", "gt"], prob=0.5),
        mtr.RandGaussianNoised(keys=["image"], prob=0.5, mean=0, std=0.1),
        mtr.RandRicianNoised(keys=["image"], prob=0.2, mean=0, std=0.2),
        mtr.RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
        mtr.RandCropByLabelClassesd(
            keys=["image", "gt"],
            label_key="gt",
            spatial_size=patch_size,
            ratios=[1, 1, 1],
            num_classes=3,
            num_samples=samples_per_volume)
    ])

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'gt']),
        mtr.EnsureChannelFirstd(keys=["image", "gt"]),
        mtr.NormalizeIntensityd(keys=["image"])
    ])

    ### General Parameters ###
    experiment_name = f"H_PRE_Exp1h_F{fold}_V3_Multi"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': True}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0, 1, 2, 3]
    logger = None
    callbacks = None
    max_epochs = 501
    precision = "bf16-mixed"
    strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = runet.default_pre_config()
    config['use_sigmoid'] = False
    model = runet.RUNet(**config)
    # checkpoints = {1: 'epoch=484_dice.ckpt', 2: 'epoch=388_dice.ckpt', 3: 'epoch=366_dice.ckpt', 4: 'epoch=464_dice.ckpt', 5: 'epoch=390_dice.ckpt'}
    # checkpoint_path = p.checkpoints_path / f"H_PRE_Exp1g_F{fold}" / checkpoints[int(fold)]
    # model = load_checkpoint(model, checkpoint_path, mode)

    # checkpoints = {1: 'epoch=174_dice.ckpt', 2: 'epoch=173_dice.ckpt', 3: 'epoch=169_dice.ckpt', 4: 'epoch=173_dice.ckpt', 5: 'epoch=178_dice.ckpt'}
    checkpoints = {1: 'epoch=182_dice.ckpt', 2: 'epoch=151_dice.ckpt', 3: 'epoch=181_dice.ckpt', 4: 'epoch=166_dice.ckpt', 5: 'epoch=171_dice.ckpt'}
    checkpoint_path = p.checkpoints_path / f"H_PRE_Exp1h_F{fold}_V2" / checkpoints[int(fold)]
    model = load_checkpoint(model, checkpoint_path, mode)

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













def exp_1a_multi(fold=1):
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
    num_workers = 16
    training_cache_rate = 0.25
    validation_cache_rate = 1.0
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
    experiment_name = f"H_PRE_Exp1a_F{fold}_Multi"
    learning_rate = 0.001
    save_step = 20
    to_load_checkpoint_path = None
    number_of_images_to_log = 3
    objective_function = cf.dice_focal_loss_monai
    objective_function_params = {'sigmoid': True, 'include_background': False}
    optimizer_weight_decay = 0.005
    echo = True

    accelerator = 'gpu'
    devices = [0, 1, 2, 3]
    logger = None
    callbacks = None
    max_epochs = 501
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
