import json
import glob
import os
from tqdm.auto import tqdm

oris = ['left', 'right', 'lower', 'upper']
# CLASS_LABEL 
#       LEFT: 0
#       RIGHT: 1
#       LOWER: 2
#       UPPER: 3
label = {"labels":[]}
paths = [glob.glob(f'/intraoral/stylegan2-ada-pytorch/data/train/{ori}/*.png') for ori in oris]
for label_idx, path_list in enumerate(paths):
    print(f"{oris[label_idx]} starts! "+"="*20+'\n')
    for path in tqdm(path_list):
        name = os.path.split(path)[-1]
        # label_ele = [os.path.join('train', oris[label_idx], name), label_idx]
        label_ele = [os.path.join(oris[label_idx], name), label_idx]
        label['labels'].append(label_ele)

with open('/intraoral/stylegan2-ada-pytorch/data/train/dataset.json', 'w') as f:
    json.dump(label, f)
