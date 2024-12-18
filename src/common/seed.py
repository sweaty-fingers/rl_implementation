import os
import random
import numpy as np
import torch

from typing import Optional

def set_seed(seed, env: Optional[gym.Env] = None):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)