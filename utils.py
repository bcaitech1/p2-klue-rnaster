import random

import numpy as np
import torch


def fix_random_state(random_seed=1818):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    return


def send_inputs_to_gpu(inputs):
    return [inp.cuda() for inp in inputs]
