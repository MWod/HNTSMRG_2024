### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable, Any, Callable
import pathlib
import time
import math
import psutil

### External Imports ###
import numpy as np
import pandas as pd
import torch as tc
import lightning as pl
import torchvision.transforms as transforms
import PIL
import matplotlib.pyplot as plt

### MONAI Imports ###
from monai.data import CacheDataset, list_data_collate, DataLoader
from monai.networks import utils as mon_utils
from monai.inferers import sliding_window_inference

### Internal Imports ###
from visualization import volumetric as vol
from evaluation import evaluation_functions as evf
from helpers import utils as u
from helpers import cost_functions as cf

########################



class LightningModule(pl.LightningModule):
    def __init__(self, training_params : dict, lightning_params : dict):
        super().__init__()
        ### Models ###
        self.model : tc.nn.Module = training_params['model']
        ### General Params ###
        self.training_params = training_params
        self.learning_rate : float = training_params['learning_rate']
        self.optimizer_weight_decay : float = training_params['optimizer_weight_decay']
        self.log_image_iters : Iterable[int] = training_params['log_image_iters']
        self.number_of_images_to_log : int = training_params['number_of_images_to_log']
        self.echo : bool = training_params['echo']
        ### Objective Functions ###
        self.objective_function : Callable = training_params['objective_function']
        self.objective_function_params : dict = training_params['objective_function_params']

    def forward(self, x):
        output = self.model(x)
        return output
    
    def configure_optimizers(self):
        optimizer = tc.optim.AdamW(self.model.parameters(), self.learning_rate, weight_decay=self.optimizer_weight_decay)
        scheduler = {
            "scheduler": tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=10),
            "frequency": 1,
            "interval": "epoch",
            "monitor": "Loss/Validation/loss",
        }
        dict = {'optimizer': optimizer, "lr_scheduler": scheduler}
        return dict
    
    def training_step(self, batch, batch_idx):
        ### Get Batch ###
        input_data, ground_truth = batch["image"], batch['gt']
        output = self.model(input_data)

        num_classes = output.shape[1]
        one_hot_gt = mon_utils.one_hot(ground_truth, num_classes=num_classes)
        loss = self.objective_function(output, one_hot_gt, **self.objective_function_params)
        dice = cf.dice_loss_monai(output, one_hot_gt, include_background=False, sigmoid=True)
        
        ### Log Losses ###
        self.log("Loss/Training/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Training/dice", dice, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        if self.current_epoch in self.log_image_iters and batch_idx < self.number_of_images_to_log:
            self.log_images(input_data, output, ground_truth, batch_idx, "Training")
        return loss
                        
    def validation_step(self, batch, batch_idx):
        ### Get Batch ###
        input_data, ground_truth = batch["image"], batch['gt']
        if input_data.shape[2] * input_data.shape[3] * input_data.shape[4] > 300_000_000:
            return
        output = sliding_window_inference(input_data, self.training_params['patch_size'], self.training_params['batch_size'], self.forward, mode='constant')
        num_classes = output.shape[1]
        one_hot_gt = mon_utils.one_hot(ground_truth, num_classes=num_classes)
        loss = self.objective_function(output, one_hot_gt, **self.objective_function_params)
        dice = cf.dice_loss_monai(output, one_hot_gt, include_background=False, sigmoid=True)
        ### Log Losses ###
        self.log("Loss/Validation/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("Loss/Validation/dice", dice, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        if self.current_epoch in self.log_image_iters and batch_idx < self.number_of_images_to_log:
            self.log_images(input_data, output, ground_truth, batch_idx, "Validation")
 
    def apply_threshold(self, output):
        output = (tc.nn.Sigmoid()(output) > 0.5).type(self.dtype)
        return output
    
    def calculate_evaluation_dice(self, output, ground_truth):
        num_classes = output.shape[1]
        one_hot_gt = mon_utils.one_hot(ground_truth, num_classes=num_classes)
        dice = evf.multiclass_dice_coefficient(output[0, :, :, :, :].detach().cpu().numpy(), one_hot_gt[0, :, :, :, :].detach().cpu().numpy())
        return dice

    def log_images(self, input_data, output, ground_truth, i, mode) -> None:
        output = self.apply_threshold(output)
        dice = self.calculate_evaluation_dice(output, ground_truth)
        spacing = [(1.0, 1.0, 1.0)]
        mt_output = u.from_one_hot(output, num_classes=output.shape[1])
        buf = vol.show_volumes_2d(input_data[0], ground_truth[0], mt_output[0], spacing=spacing[0], return_buffer=True, suptitle=f"Dice: {dice}", names=["Input", "Ground-Truth", "Output"], dpi=100, show=False)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image).unsqueeze(0)[0]
        title = f"{mode}. Case: {i}, Epoch: {str(self.current_epoch)} Dice: {dice}"
        self.logger.experiment.add_image(title, image, 0)
        plt.close('all')

class LightningDataModule(pl.LightningDataModule):
    def __init__(self,
                training_data,
                validation_data,
                training_transforms,
                validation_transforms,
                training_cache_rate,
                validation_cache_rate,
                num_workers,
                batch_size):
        super().__init__()
        self.training_data = training_data
        self.validation_data = validation_data
        self.training_transforms = training_transforms
        self.validation_transforms = validation_transforms
        self.training_cache_rate = training_cache_rate
        self.validation_cache_rate = validation_cache_rate
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.training_dataset = CacheDataset(data=self.training_data, transform=self.training_transforms, cache_rate=self.training_cache_rate, num_workers=self.num_workers)
        self.validation_dataset = CacheDataset(data=self.validation_data, transform=self.validation_transforms, cache_rate=self.validation_cache_rate, num_workers=self.num_workers)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=list_data_collate, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=1, num_workers=self.num_workers, collate_fn=list_data_collate)

class SegmentationTrainer():
    def __init__(self, **training_params : dict):
        ### General params
        self.training_params = training_params
        self.lightning_params = training_params['lightning_params']
        self.checkpoints_path : Union[str, pathlib.Path] = training_params['checkpoints_path']
        self.to_load_checkpoint_path : Union[str, pathlib.Path, None] = training_params['to_load_checkpoint_path']
        if self.to_load_checkpoint_path is None:
            self.module = LightningModule(self.training_params, self.lightning_params)
        else:
            self.load_checkpoint()
    
        self.trainer = pl.Trainer(**self.lightning_params)
        self.data_module = LightningDataModule(
                training_params['training_data'],
                training_params['validation_data'],
                training_params['training_transforms'],
                training_params['validation_transforms'],
                training_params['training_cache_rate'],
                training_params['validation_cache_rate'],
                training_params['num_workers'],
                training_params['batch_size'],
        )

    def save_checkpoint(self) -> None:
        self.trainer.save_checkpoint(pathlib.Path(self.checkpoints_path) / "Last_Iteration")

    def load_checkpoint(self) -> None:
        self.module = LightningModule.load_from_checkpoint(self.to_load_checkpoint_path, training_params=self.training_params, lightning_params=self.lightning_params) 

    def run(self) -> None:
        self.trainer.fit(self.module, self.data_module)
        self.save_checkpoint()