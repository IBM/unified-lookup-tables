__copyright__ = """LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2025 - AI4SD
ALL RIGHTS RESERVED
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int):
    """Seed everything to make the run reproducible"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True