from PIL import Image
from glob import glob
from tqdm.auto import tqdm
import numpy as np
import os

oris = ['left', 'right', 'lower', 'upper']
# CLASS_LABEL 
#       LEFT: 0
#       RIGHT: 1
#       LOWER: 2
#       UPPER: 3

paths = [glob(f'/intraoral/Intraoral-i2i-dataset/fr-{ori}/train/*.png') for ori in oris]
for label_idx, path_list in enumerate(paths):
    print(f"{oris[label_idx]} starts! "+"="*20+'\n')
    if label_idx == 0 or label_idx == 1:
        continue
    for path in tqdm(path_list):
        img = np.split(np.array(Image.open(path)), 2, axis=1)[-1]
        name = os.path.split(path)[-1]
        img = Image.fromarray(img).resize((256, 256))
        img.save(f'/intraoral/stylegan2-ada-pytorch/data/train/{oris[label_idx]}/'+name)