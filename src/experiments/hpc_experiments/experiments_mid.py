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
from datasets import dataset as ds
from networks import swinunetr, unetr, attentionunet
########################

def load_checkpoint(model, checkpoint_path, mode):
    if mode == "pre":
        state_dict = tc.load(checkpoint_path)['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict[key.replace("model.", "", 1)] = state_dict[key]
        model.load_state_dict(new_state_dict)
    elif mode == "mid":
        state_dict = tc.load(checkpoint_path)['state_dict']
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
    mode = 'mid'

    ### Prepare Data ###
    training_data = ds.prepare_data(training_dataset_path, training_csv_path, mode)
    validation_data = ds.prepare_data(validation_dataset_path, validation_csv_path, mode)

    ### Prepare Loading & Augmentation ###
    samples_per_volume = 4
    batch_size = 1
    patch_size = (128, 128, 128)
    num_workers = 4
    training_cache_rate = 0.05
    validation_cache_rate = 0.05

    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
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
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
    ])


    ### General Parameters ###
    experiment_name = f"H_MID_Exp1_F{fold}"
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
    config = runet.default_mid_config()
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
    mode = 'mid'

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
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
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
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
    ])


    ### General Parameters ###
    experiment_name = f"H_MID_Exp1a_F{fold}_2"
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
    precision = "bf16-mixed"
    # strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = runet.default_mid_config()
    config['use_sigmoid'] = False
    model = runet.RUNet(**config)
    checkpoint_path = p.checkpoints_path / f"H_MID_Exp1a_F{fold}" / 'epoch=139_general.ckpt'
    model = load_checkpoint(model, checkpoint_path, mode)

    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['precision'] = precision
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



def exp_1b(fold=1):
    """
    """
    ### Dataset Params ###
    training_dataset_path = p.parsed_data_path / "Spacing_05_05_05"
    validation_dataset_path = p.parsed_data_path / "Spacing_05_05_05"
    training_csv_path = p.csv_path / f"training_fold_{fold}.csv"
    validation_csv_path = p.csv_path / f"val_fold_{fold}.csv"
    mode = 'mid'

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
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.RandAxisFlipd(keys=['image', 'prevolume', 'pregt', 'gt'], prob=0.5),
        mtr.RandZoomd(keys=['image', 'prevolume', 'pregt', 'gt'], prob=0.5, min_zoom=0.7, max_zoom=1.3, mode=['area', 'area', 'nearest', 'nearest'], padding_mode=['constant', 'constant', 'constant', 'constant']),
        mtr.RandGaussianNoised(keys=["image", "prevolume"], prob=0.5, mean=0, std=0.1),
        mtr.RandRicianNoised(keys=["image", 'prevolume'], prob=0.2, mean=0, std=0.2),
        mtr.RandGaussianSmoothd(keys=["image", "prevolume"], prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
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
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
    ])


    ### General Parameters ###
    experiment_name = f"H_MID_Exp1b_F{fold}"
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
    precision = "bf16-mixed"
    # strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = runet.default_mid_config()
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






def exp_1c(fold=1):
    """
    """
    ### Dataset Params ###
    training_dataset_path = p.parsed_data_path / "Spacing_05_05_05"
    validation_dataset_path = p.parsed_data_path / "Spacing_05_05_05"
    training_csv_path = p.csv_path / f"training_fold_{fold}.csv"
    validation_csv_path = p.csv_path / f"val_fold_{fold}.csv"
    mode = 'mid'

    ### Prepare Data ###
    training_data = ds.prepare_data(training_dataset_path, training_csv_path, mode)
    validation_data = ds.prepare_data(validation_dataset_path, validation_csv_path, mode)

    ### Prepare Loading & Augmentation ###
    samples_per_volume = 4
    batch_size = 1
    patch_size = (128, 128, 128)
    num_workers = 12
    training_cache_rate = 0.15
    validation_cache_rate = 1.0

    # training_transforms = mtr.Compose([
    #     mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
    #     mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
    #     mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
    #     mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
    #     mtr.RandCropByLabelClassesd(
    #         keys=["image", "gt"],
    #         label_key="gt",
    #         spatial_size=patch_size,
    #         ratios=[1, 1, 1],
    #         num_classes=3,
    #         num_samples=samples_per_volume
    #     )
    # ])
    
    
    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.RandAxisFlipd(keys=['image', 'prevolume', 'pregt', 'gt'], prob=0.5),
        mtr.RandGaussianNoised(keys=['image', 'prevolume'], prob=0.5, mean=0, std=0.1),
        mtr.RandRicianNoised(keys=['image', 'prevolume'], prob=0.2, mean=0, std=0.2),
        mtr.RandGaussianSmoothd(keys=['image', 'prevolume'], prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
        mtr.RandCropByLabelClassesd(
            keys=["image", "gt"],
            label_key="gt",
            spatial_size=patch_size,
            ratios=[1, 1, 1],
            num_classes=3,
            num_samples=samples_per_volume
        )
    ])

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
    ])


    ### General Parameters ###
    experiment_name = f"H_MID_Exp1c_F{fold}_V3"
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
    precision = "bf16-mixed"
    # strategy = "ddp"
    # accumulate_grad_batches = 16
    deterministic = False

    ### Declare Model ###
    config = runet.default_mid_config()
    config['use_sigmoid'] = False
    model = runet.RUNet(**config)
    # checkpoints = {1: 'epoch=81_dice.ckpt', 2: 'epoch=82_dice.ckpt', 3: 'epoch=79_general.ckpt', 4: 'epoch=79_general.ckpt', 5: 'epoch=79_general.ckpt'}
    # checkpoint_path = p.checkpoints_path / f"H_MID_Exp1c_F{fold}" / checkpoints[int(fold)]
    
    checkpoints = {1: 'epoch=120_dice.ckpt', 2: 'epoch=61_dice.ckpt', 3: 'epoch=126_dice.ckpt', 4: 'epoch=68_dice.ckpt', 5: 'epoch=117_dice.ckpt'}
    checkpoint_path = p.checkpoints_path / f"H_MID_Exp1c_F{fold}_V2" / checkpoints[int(fold)]
    model = load_checkpoint(model, checkpoint_path, mode)

    ### Lightning Parameters ###
    lighting_params = dict()
    lighting_params['accelerator'] = accelerator
    lighting_params['devices'] = devices
    lighting_params['logger'] = logger
    lighting_params['callbacks'] = callbacks
    lighting_params['max_epochs'] = max_epochs
    lighting_params['precision'] = precision
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
    mode = 'mid'

    ### Prepare Data ###
    training_data = ds.prepare_data(training_dataset_path, training_csv_path, mode)
    validation_data = ds.prepare_data(validation_dataset_path, validation_csv_path, mode)

    ### Prepare Loading & Augmentation ###
    samples_per_volume = 4
    batch_size = 1
    patch_size = (128, 128, 128)
    num_workers = 12
    training_cache_rate = 0.15
    validation_cache_rate = 1.0

    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
        mtr.RandCropByLabelClassesd(
            keys=["image", "gt"],
            label_key="gt",
            spatial_size=patch_size,
            ratios=[1, 1, 1],
            num_classes=3,
            num_samples=samples_per_volume
        )
    ])

    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
    ])

    ### General Parameters ###
    experiment_name = f"H_MID_Exp2a_F{fold}_V2"
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
    config = unetr.default_mid_config()
    model = unetr.UNETR(**config)
    checkpoints = {1: 'epoch=68_dice.ckpt'}
    checkpoint_path = p.checkpoints_path / f"H_MID_Exp2a_F{fold}" / checkpoints[int(fold)]
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
    mode = 'mid'

    ### Prepare Data ###
    training_data = ds.prepare_data(training_dataset_path, training_csv_path, mode)
    validation_data = ds.prepare_data(validation_dataset_path, validation_csv_path, mode)

    ### Prepare Loading & Augmentation ###
    samples_per_volume = 4
    batch_size = 1
    patch_size = (128, 128, 128)
    num_workers = 12
    training_cache_rate = 0.15
    validation_cache_rate = 1.0

    training_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
        mtr.RandCropByLabelClassesd(
            keys=["image", "gt"],
            label_key="gt",
            spatial_size=patch_size,
            ratios=[1, 1, 1],
            num_classes=3,
            num_samples=samples_per_volume
        )
    ])


    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
    ])

    ### General Parameters ###
    experiment_name = f"H_MID_Exp2b_F{fold}_V2"
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
    config = attentionunet.default_mid_config()
    model = attentionunet.AttentionUNet(**config)
    checkpoints = {1: 'epoch=88_dice.ckpt'}
    checkpoint_path = p.checkpoints_path / f"H_MID_Exp2b_F{fold}" / checkpoints[int(fold)]
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
    mode = 'mid'

    ### Prepare Data ###
    training_data = ds.prepare_data(training_dataset_path, training_csv_path, mode)
    validation_data = ds.prepare_data(validation_dataset_path, validation_csv_path, mode)

    ### Prepare Loading & Augmentation ###
    samples_per_volume = 4
    batch_size = 1
    patch_size = (128, 128, 128)
    num_workers = 12
    training_cache_rate = 0.15
    validation_cache_rate = 1.0

    if int(fold) == 1:
        training_transforms = mtr.Compose([
            mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
            mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
            mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
            mtr.RandAxisFlipd(keys=['image', 'prevolume', 'pregt', 'gt'], prob=0.5),
            mtr.RandGaussianNoised(keys=['image', 'prevolume'], prob=0.5, mean=0, std=0.1),
            mtr.RandRicianNoised(keys=['image', 'prevolume'], prob=0.2, mean=0, std=0.2),
            mtr.RandGaussianSmoothd(keys=['image', 'prevolume'], prob=0.2, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
            mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
            mtr.RandCropByLabelClassesd(
                keys=["image", "gt"],
                label_key="gt",
                spatial_size=patch_size,
                ratios=[1, 1, 1],
                num_classes=3,
                num_samples=samples_per_volume
            )
        ])
    else:
        training_transforms = mtr.Compose([
            mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
            mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
            mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
            mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
            mtr.RandCropByLabelClassesd(
                keys=["image", "gt"],
                label_key="gt",
                spatial_size=patch_size,
                ratios=[1, 1, 1],
                num_classes=3,
                num_samples=samples_per_volume
            )
        ])


    validation_transforms = mtr.Compose([
        mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
        mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
        mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
    ])

    ### General Parameters ###
    if int(fold) == 1:
        experiment_name = f"H_MID_Exp2c_F{fold}_V3"
    else:
        experiment_name = f"H_MID_Exp2c_F{fold}"
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
    config = swinunetr.default_mid_config()
    model = swinunetr.SwinUNetR(**config)
    checkpoints = {1: 'epoch=100_dice.ckpt'}
    
    if int(fold) == 1:
        checkpoint_path = p.checkpoints_path / f"H_MID_Exp2c_F{fold}_V2" / checkpoints[int(fold)]
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
