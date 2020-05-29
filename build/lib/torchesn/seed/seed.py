# This is a util function to manuall set a seed that should work for CPU or GPU runs of pytorch-esn

import random
import torch
import numpy as np
import datetime
import math

# Define a manual seed setting function

def set_seed(device, manualSeed=None):
     # if manualSeed is None:
     #     manualSeed = randint(1, 200)
     np.random.seed(manualSeed)
     random.seed(manualSeed)
     torch.manual_seed(manualSeed)
     # if you are using GPU
     if device == 'GPU':
          torch.cuda.manual_seed(manualSeed)
          torch.cuda.manual_seed_all(manualSeed)
          torch.backends.cudnn.enabled = False
          torch.backends.cudnn.benchmark = False
          torch.backends.cudnn.deterministic = True
     def _init_fn():
         np.random.seed(manualSeed)
     return manualSeed
