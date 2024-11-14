### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
from scipy.ndimage import morphology
import SimpleITK as sitk

from monai import transforms as mtr
from monai.data import CacheDataset, list_data_collate, DataLoader
from monai.inferers import sliding_window_inference


### Internal Imports ###
from paths import pc_paths as p
from datasets import dataset as ds
from networks import runet, swinunetr
from evaluation import np_metrics as npm
from evaluation import tc_metrics as tcm

########################


def run_evaluation(dataset_path, csv_path, output_csv_path, output_path, model, mode, device="cuda:0", thresholds=(0.5, 0.5)):
    data = ds.prepare_data(dataset_path, csv_path, mode)
    batch_size = 1
    num_workers = 8
    patch_size = (128, 128, 128)
    model = model.to(device)
    if mode == "pre":
        transforms = mtr.Compose([
            mtr.LoadImaged(keys=['image', 'gt']),
            mtr.EnsureChannelFirstd(keys=["image", "gt"]),
            mtr.NormalizeIntensityd(keys=["image"])
        ])
    else:
        transforms = mtr.Compose([
            mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
            mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
            mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
            mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
        ])

    cache_rate = 0.0
    dataset = CacheDataset(data=data, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
    dataloder = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=list_data_collate, shuffle=False)
    print(f"Dataloader length: {len(dataloder)}")

    outputs = []
    numerator_1 = 0.0
    numerator_2 = 0.0
    denominator_1 = 0.0
    denominator_2 = 0.0

    def calculate_metric(input1, input2):
        input1 = np.atleast_1d(input1.astype(bool))
        input2 = np.atleast_1d(input2.astype(bool))
        intersection = 2 * np.count_nonzero(input1 & input2)
        size_i1 = np.count_nonzero(input1)
        size_i2 = np.count_nonzero(input2)
        return intersection, size_i1 + size_i2

    with tc.no_grad():
        for idx, batch in enumerate(dataloder):
            input_data, ground_truth = batch["image"], batch['gt']
            input_data = input_data.to(device)
            output = sliding_window_inference(input_data, patch_size, batch_size, model.forward, mode='constant')
            output = tc.argmax(tc.sigmoid(output), dim=1, keepdim=True)

            gt_class_1 = (ground_truth == 1).detach().cpu().numpy()[0, 0]
            gt_class_2 = (ground_truth == 2).detach().cpu().numpy()[0, 0]
            output_class_1 = (output == 1).detach().cpu().numpy()[0, 0]
            output_class_2 = (output == 2).detach().cpu().numpy()[0, 0]
            print(f"Output shape: {output_class_1.shape}")
            print(f"GT Shape: {gt_class_1.shape}")

            dice_1 = npm.dc(output_class_1, gt_class_1) 
            dice_2 = npm.dc(output_class_2, gt_class_2) 

            n1, d1 = calculate_metric(gt_class_1, output_class_1)
            n2, d2 = calculate_metric(gt_class_2, output_class_2)

            numerator_1 += n1
            denominator_1 += d1

            numerator_2 += n2
            denominator_2 += d2

            dice_1_agg = numerator_1 / denominator_1
            dice_2_agg = numerator_2 / denominator_2

            print(f"Current case: {idx}")
            print(f"Dice 1: {dice_1}")
            print(f"Dice 2: {dice_2}")
            print(f"Dice 1 Agg: {dice_1_agg}")
            print(f"Dice 2 Agg: {dice_2_agg}")
            print()
            
            output_folder = output_path / str(idx)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
                input = sitk.GetImageFromArray(input_data.detach().cpu().numpy()[0, 0])
                output = sitk.GetImageFromArray(output.detach().cpu().numpy()[0, 0])
                gt = sitk.GetImageFromArray(ground_truth.detach().cpu().numpy()[0, 0])
                sitk.WriteImage(input, str(output_folder / "image.mha"))
                sitk.WriteImage(output, str(output_folder / "output.nrrd"), useCompression=True)
                sitk.WriteImage(gt, str(output_folder / "gt.nrrd"), useCompression=True)


            to_append = (idx, dice_1, dice_2, dice_1_agg, dice_2_agg)
            outputs.append(to_append)

    dataframe = pd.DataFrame(outputs, columns=['SubjectID', 'Dice_1', 'Dice_2', 'Dice_Agg_1', 'Dice_Agg_2'])
    dataframe.to_csv(output_csv_path, index=False)





def run_evaluations(dataset_path, csv_path, output_csv_path, output_path, models, mode, device="cuda:0"):
    data = ds.prepare_data(dataset_path, csv_path, mode)
    batch_size = 1
    num_workers = 8
    patch_size = (128, 128, 128)
    if mode == "pre":
        transforms = mtr.Compose([
            mtr.LoadImaged(keys=['image', 'gt']),
            mtr.EnsureChannelFirstd(keys=["image", "gt"]),
            mtr.NormalizeIntensityd(keys=["image"])
        ])
    else:
        transforms = mtr.Compose([
            mtr.LoadImaged(keys=['image', 'prevolume', 'pregt', 'gt']),
            mtr.EnsureChannelFirstd(keys=['image', 'prevolume', 'pregt', 'gt']),
            mtr.NormalizeIntensityd(keys=["image", "prevolume"]),
            mtr.ConcatItemsd(keys=['image', 'prevolume', 'pregt'], name='image'),
        ])

    cache_rate = 0.0
    dataset = CacheDataset(data=data, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
    dataloder = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=list_data_collate, shuffle=False)
    print(f"Dataloader length: {len(dataloder)}")

    outputs = []
    numerator_1 = 0.0
    numerator_2 = 0.0
    denominator_1 = 0.0
    denominator_2 = 0.0

    def calculate_metric(input1, input2):
        input1 = np.atleast_1d(input1.astype(bool))
        input2 = np.atleast_1d(input2.astype(bool))
        intersection = 2 * np.count_nonzero(input1 & input2)
        size_i1 = np.count_nonzero(input1)
        size_i2 = np.count_nonzero(input2)
        return intersection, size_i1 + size_i2

    with tc.no_grad():
        for idx, batch in enumerate(dataloder):
            input_data, ground_truth = batch["image"], batch['gt']
            input_data = input_data.to(device)
            for midx, model in enumerate(models):
                if midx == 0:
                    output = tc.sigmoid(sliding_window_inference(input_data, patch_size, batch_size, model.forward, mode='constant'))
                else:
                    output += tc.sigmoid(sliding_window_inference(input_data, patch_size, batch_size, model.forward, mode='constant'))
            output = tc.argmax(output, dim=1, keepdim=True)

            gt_class_1 = (ground_truth == 1).detach().cpu().numpy()[0, 0]
            gt_class_2 = (ground_truth == 2).detach().cpu().numpy()[0, 0]
            output_class_1 = (output == 1).detach().cpu().numpy()[0, 0]
            output_class_2 = (output == 2).detach().cpu().numpy()[0, 0]
            print(f"Output shape: {output_class_1.shape}")
            print(f"GT Shape: {gt_class_1.shape}")

            dice_1 = npm.dc(output_class_1, gt_class_1) 
            dice_2 = npm.dc(output_class_2, gt_class_2) 

            n1, d1 = calculate_metric(gt_class_1, output_class_1)
            n2, d2 = calculate_metric(gt_class_2, output_class_2)

            numerator_1 += n1
            denominator_1 += d1

            numerator_2 += n2
            denominator_2 += d2

            dice_1_agg = numerator_1 / denominator_1
            dice_2_agg = numerator_2 / denominator_2

            print(f"Current case: {idx}")
            print(f"Dice 1: {dice_1}")
            print(f"Dice 2: {dice_2}")
            print(f"Dice 1 Agg: {dice_1_agg}")
            print(f"Dice 2 Agg: {dice_2_agg}")
            print()
            
            output_folder = output_path / str(idx)
            if not os.path.isdir(output_folder) and idx == 2:
                os.makedirs(output_folder)
                input = sitk.GetImageFromArray(input_data.detach().cpu().numpy()[0, 0])
                output = sitk.GetImageFromArray(output.detach().cpu().numpy()[0, 0])
                gt = sitk.GetImageFromArray(ground_truth.detach().cpu().numpy()[0, 0])
                sitk.WriteImage(input, str(output_folder / "image.mha"))
                sitk.WriteImage(output, str(output_folder / "output.nrrd"), useCompression=True)
                sitk.WriteImage(gt, str(output_folder / "gt.nrrd"), useCompression=True)

            to_append = (idx, dice_1, dice_2, dice_1_agg, dice_2_agg)
            outputs.append(to_append)

    dataframe = pd.DataFrame(outputs, columns=['SubjectID', 'Dice_1', 'Dice_2', 'Dice_Agg_1', 'Dice_Agg_2'])
    dataframe.to_csv(output_csv_path, index=False)








def load_model(checkpoint_path, mode):
    if mode == "pre":
        config = runet.default_pre_config()
        config['use_sigmoid'] = False
        model = runet.RUNet(**config)
        state_dict = tc.load(checkpoint_path)['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict[key.replace("model.", "")] = state_dict[key]
        model.load_state_dict(new_state_dict)
        model.eval()
    elif mode == "mid":
        config = runet.default_mid_config()
        config['use_sigmoid'] = False
        model = runet.RUNet(**config)
        state_dict = tc.load(checkpoint_path)['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict[key.replace("model.", "")] = state_dict[key]
        model.load_state_dict(new_state_dict)
        model.eval()
    return model

def load_swinunetr(checkpoint_path, mode):
    if mode == "pre":
        config = swinunetr.default_pre_config()
        model = swinunetr.SwinUNetR(**config)
        state_dict = tc.load(checkpoint_path)['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict[key.replace("model.", "", 1)] = state_dict[key]
        model.load_state_dict(new_state_dict)
        model.eval()
    elif mode == "mid":
        config = swinunetr.default_mid_config()
        model = swinunetr.SwinUNetR(**config)
        state_dict = tc.load(checkpoint_path)['state_dict']
        new_state_dict = {}
        for key in state_dict.keys():
            new_state_dict[key.replace("model.", "", 1)] = state_dict[key]
        model.load_state_dict(new_state_dict)
        model.eval()
    return model

def run():
    pass



if __name__ == "__main__":
    run()