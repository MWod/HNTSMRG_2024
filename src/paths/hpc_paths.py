import os
import pathlib

data_path = None # TODO
raw_data_path = data_path / "RAW" / "HNTSMRG24_train"
parsed_data_path = data_path / "Parsed"
csv_path = parsed_data_path / "CSV"

### Training Paths ###
project_path = None # TODO
checkpoints_path = project_path / "Checkpoints"
logs_path = project_path / "Logs"
figures_path = project_path / "Figures"
models_path = project_path / "Models"
results_path = project_path / "Results"