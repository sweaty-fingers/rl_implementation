import os

ROOT = "/mnt/data"

CONFIG = {
    "DIRS":{
        "ROOTS":{
            "ENV": "envs",
            "EXPERIEMNTS": os.path.join(ROOT, "experiment_logs"),
        }
    }
}