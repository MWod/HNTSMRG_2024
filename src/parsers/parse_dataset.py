### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pathlib
import shutil

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import KFold

### Internal Imports ###
from paths import pc_paths as p
from preprocessing import preprocessing_volumetric as pre_vol



def parse_hntsmrg():
    input_path = p.raw_data_path
    output_data_path = p.parsed_data_path / "Spacing_05_05_05"
    # output_data_path = p.parsed_data_path / "Spacing_075_075_075"
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    output_csv_path = p.csv_path / "dataset.csv"

    cases = os.listdir(input_path)
    print(f"Number of cases: {len(cases)}")
    print()

    device = "cuda:0"
    new_spacing = (0.5, 0.5, 0.5)
    # new_spacing = (0.75, 0.75, 0.75)

    dataframe = []
    for case in cases:
        print()
        print(f"Current case: {case}")
        midRT_path = input_path / case / "midRT"
        preRT_path = input_path / case / "preRT"

        # midRT paths
        midRT_volume_path = midRT_path / f"{case}_midRT_T2.nii.gz"
        midRT_gt_path = midRT_path / f"{case}_midRT_mask.nii.gz"
        midRT_prevolume_path = midRT_path / f"{case}_preRT_T2_registered.nii.gz"
        midRT_pregt_path = midRT_path / f"{case}_preRT_mask_registered.nii.gz"

        # preRT paths
        preRT_volume_path = preRT_path / f"{case}_preRT_T2.nii.gz"
        preRT_gt_path = preRT_path / f"{case}_preRT_mask.nii.gz"

        # Parse Cases
        preprocessed_midRT_volume = resample_case(midRT_volume_path, new_spacing, device=device, mode='bilinear')
        preprocessed_midRT_gt = resample_case(midRT_gt_path, new_spacing, device=device, mode='nearest')
        preprocessed_midRT_prevolume = resample_case(midRT_prevolume_path, new_spacing, device=device, mode='bilinear')
        preprocessed_midRT_pregt = resample_case(midRT_pregt_path, new_spacing, device=device, mode='nearest')

        preprocessed_preRT_volume = resample_case(preRT_volume_path, new_spacing, device=device, mode='bilinear')
        preprocessed_preRT_gt = resample_case(preRT_gt_path, new_spacing, device=device, mode='nearest')

        # Save Volumes
        output_midRT_path = pathlib.Path(f"{case}_midRT")
        output_preRT_path = pathlib.Path(f"{case}_preRT")

        save_midRT_volume_path = output_data_path / output_midRT_path / "midRT.nii.gz"
        save_midRT_gt_path = output_data_path / output_midRT_path / "midRT_gt.nii.gz"
        save_midRT_prevolume_path = output_data_path / output_midRT_path / "midRT_pre.nii.gz"
        save_midRT_pregt_path = output_data_path / output_midRT_path / "midRT_pregt.nii.gz"

        save_preRT_volume_path = output_data_path / output_preRT_path / "preRT.nii.gz"
        save_preRT_gt_path = output_data_path / output_preRT_path / "preRT_gt.nii.gz"

        if not os.path.exists(os.path.dirname(save_midRT_volume_path)):
            os.makedirs(os.path.dirname(save_midRT_volume_path))

        if not os.path.exists(os.path.dirname(save_preRT_volume_path)):
            os.makedirs(os.path.dirname(save_preRT_volume_path))

        sitk.WriteImage(preprocessed_midRT_volume, str(save_midRT_volume_path))
        sitk.WriteImage(preprocessed_midRT_gt, str(save_midRT_gt_path), useCompression=True)
        sitk.WriteImage(preprocessed_midRT_prevolume, str(save_midRT_prevolume_path))
        sitk.WriteImage(preprocessed_midRT_pregt, str(save_midRT_pregt_path), useCompression=True)

        sitk.WriteImage(preprocessed_preRT_volume, str(save_preRT_volume_path))
        sitk.WriteImage(preprocessed_preRT_gt, str(save_preRT_gt_path), useCompression=True)

        to_append = (output_preRT_path, output_midRT_path)
        dataframe.append(to_append)

    # dataframe = pd.DataFrame(dataframe, columns=['preRT Path', 'midRT Path'])
    # dataframe.to_csv(output_csv_path, index=False)


def resample_case(volume_path, new_spacing, device="cpu", mode='bilinear'):
    volume = sitk.ReadImage(volume_path)
    spacing = volume.GetSpacing()
    volume = sitk.GetArrayFromImage(volume).swapaxes(0, 1).swapaxes(1, 2)
    print(f"Volume shape: {volume.shape}")
    print(f"Spacing: {spacing}")
    volume_tc = tc.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    print(f"Volume TC shape: {volume_tc.shape}")
    resampled_volume_tc = pre_vol.resample_to_spacing_tc(volume_tc, spacing, new_spacing, mode=mode)
    print(f"Resampled Volume TC shape: {resampled_volume_tc.shape}")
    if mode == 'bilinear':
        resampled_volume = resampled_volume_tc[0, 0, :, :, :].detach().cpu().numpy().astype(np.float32)
    else:
        resampled_volume = resampled_volume_tc[0, 0, :, :, :].detach().cpu().numpy()
    resampled_volume = sitk.GetImageFromArray(resampled_volume.swapaxes(2, 1).swapaxes(1, 0))
    resampled_volume.SetSpacing(new_spacing)
    print(f"Output spacing: {resampled_volume.GetSpacing()}")
    print(f"Output size: f{resampled_volume.GetSize()}")
    return resampled_volume


def split_dataframe(num_folds=5, seed=1234):
    input_csv_path = p.csv_path / "dataset.csv"
    output_splits_path = p.csv_path
    if not os.path.isdir(os.path.dirname(output_splits_path)):
        os.makedirs(os.path.dirname(output_splits_path))
    dataframe = pd.read_csv(input_csv_path)
    print(f"Dataset size: {len(dataframe)}")
    kf = KFold(n_splits=num_folds, shuffle=True)
    folds = kf.split(dataframe)
    for fold in range(num_folds):
        train_index, test_index = next(folds)
        current_training_dataframe = dataframe.loc[train_index]
        current_validation_dataframe = dataframe.loc[test_index]
        print(f"Fold {fold + 1} Training dataset size: {len(current_training_dataframe)}")
        print(f"Fold {fold + 1} Validation dataset size: {len(current_validation_dataframe)}")
        training_csv_path = output_splits_path / f"training_fold_{fold+1}.csv"
        validation_csv_path = output_splits_path / f"val_fold_{fold+1}.csv"
        current_training_dataframe.to_csv(training_csv_path, index=False)
        current_validation_dataframe.to_csv(validation_csv_path, index=False)


def run():
    # parse_hntsmrg()
    # split_dataframe()
    pass


if __name__ == "__main__":
    run()