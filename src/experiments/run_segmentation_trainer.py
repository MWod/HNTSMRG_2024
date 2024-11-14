### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import pathlib

### External Imports ###
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

### Internal Imports ###
from paths import pc_paths as p
from training import segmentation_trainer as sg
from experiments.pc_experiments import experiments as exp
from experiments.pc_experiments import experiments_pre as exp_pre
########################


def initialize(training_params):
    experiment_name = training_params['experiment_name']
    num_iterations = training_params['num_iterations']
    save_step = training_params['save_step']
    checkpoints_path = os.path.join(p.checkpoints_path, experiment_name)
    log_image_iters = list(range(0, num_iterations, save_step))
    pathlib.Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    log_dir = os.path.join(p.logs_path, experiment_name)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    general_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_general', every_n_epochs=save_step, save_top_k=-1)
    best_loss_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_loss', save_top_k=1, mode='min', monitor="Loss/Validation/loss")
    best_dice_checkpoint = ModelCheckpoint(dirpath=checkpoints_path, filename='{epoch}_dice', save_top_k=1, mode='min', monitor="Loss/Validation/dice")
    logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=experiment_name)
    training_params['lightning_params']['logger'] = logger
    training_params['lightning_params']['callbacks'] = [general_checkpoint, best_loss_checkpoint, best_dice_checkpoint]   
    training_params['checkpoints_path'] = checkpoints_path
    training_params['log_image_iters'] = log_image_iters
    return training_params

def run_training(training_params):
    training_params = initialize(training_params)
    trainer = sg.SegmentationTrainer(**training_params)
    trainer.run()


if __name__ == "__main__":
    # run_training(exp.test_lightning_monai_single_gpu())
    # run_training(exp.test_lightning_monai_two_gpus())
    # run_training(exp.test_lightning_monai_single_gpu_mid())

    # for fold in range(1, 6):
    #     run_training(exp_pre.exp_1(fold))

    run_training(exp_pre.exp_2(fold=1))