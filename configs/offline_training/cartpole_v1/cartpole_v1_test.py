import os
from configs.offline_training.shared import CONFIG

root_data_dir = CONFIG["dirs"]["root"]["dataset"]

training_info = {"processed_dataname": "cartpole_v1_test"}

DIRS = {
    "processed_dataset":{
        "csv": os.path.join(root_data_dir, training_info["processed_dataname"], "csv"),
        "pkl": os.path.join(root_data_dir, training_info["processed_dataname"], "pkl")
        }
    }
CONFIG["training_info"] = training_info
CONFIG["dirs"]["processed_dataset"] = DIRS["processed_dataset"]
