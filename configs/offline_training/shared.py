import os

ROOT = "/mnt/data"
CONFIG = {
    "dirs":{
        "root":{
            "experiment_logs": os.path.join(ROOT, "experiment_logs"),
            "dataset": os.path.join(ROOT, "dataset")
        }
    },
}