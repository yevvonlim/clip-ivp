import os
import re
from typing import List, Optional
from tqdm.auto import tqdm

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import clip
import legacy

from torchvision.transforms import Compose, Resize, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

network_pkl = ''
print('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore