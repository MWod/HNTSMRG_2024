### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

### External Imports ###
import pandas as pd

### Internal Imports ###

########################



def prepare_data(dataset_path, dataframe_path, mode):
    dataframe = pd.read_csv(dataframe_path)
    dataset = []
    for idx in range(len(dataframe)):
        row = dataframe.iloc[idx]
        if mode == "pre":
            volume_path = dataset_path / row['preRT Path'] / "preRT.nii.gz"
            gt_path = dataset_path / row['preRT Path'] / "preRT_gt.nii.gz"
            to_append = {"image": volume_path, "gt": gt_path}
        elif mode == "mid":
            volume_path = dataset_path / row['midRT Path'] / "midRT.nii.gz"
            gt_path = dataset_path / row['midRT Path'] / "midRT_gt.nii.gz"
            prevolume_path = dataset_path / row['midRT Path'] / "midRT_pre.nii.gz"
            pregt_path = dataset_path / row['midRT Path'] / "midRT_pregt.nii.gz"
            to_append = {"image": volume_path, "prevolume": prevolume_path, "pregt": pregt_path, "gt": gt_path}
        dataset.append(to_append)
    return dataset