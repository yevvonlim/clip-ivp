import os
import re
from typing import List, Optional
from tqdm.auto import tqdm

import click
import dnnlib
from glob import glob
import numpy as np
from PIL import Image
import torch
import clip
import legacy
import copy

from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC




def _convert_image_to_rgb(image):
    return image.convert("RGB")


def lerp(z1, z2, n=5):
    delta = z2-z1
    lerps = []
    for i in range(n):
        lerps.append(z1 + i * delta/n)
    lerps.append(z2)
    return lerps

device = torch.device('cuda')
network_pkl = '/intraoral/stylegan2-ada-pytorch/class-conditional/vit-L-gamma5-clip/network-snapshot-007600.pkl'
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

encoder, _ = clip.load(name='ViT-L/14', device=device)
preprocess = Compose([
    Resize((224, 224), interpolation=BICUBIC),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

root = '/intraoral/stylegan2-ada-pytorch/extrapolation/'
outdir = root
resizer = Resize((180, 300))
seeds = [0, 1, 2, 3, 5, 8]
seed = 0
k=10
mode = 'z_interpolate'
truncation_psi = 0.7
alphas = np.linspace(0, 3, 3, endpoint=True)
# idx = np.random.random_integers(0, 767, 20)

sources = [(torch.from_numpy(np.array(Image.open(path).convert("RGB")).transpose(2, 0, 1)).to(device, dtype=torch.float32).unsqueeze(0), path) for path in sorted(glob(os.path.join(root, 'source/*.png')))]
before_imgs = [torch.from_numpy(np.array(Image.open(path)).transpose(2, 0, 1)).to(device, dtype=torch.float32).unsqueeze(0) for path in sorted(glob(os.path.join(root, 'orthodontic/before/*.png')))]
after_imgs = [torch.from_numpy(np.array(Image.open(path)).transpose(2, 0, 1)).to(device, dtype=torch.float32).unsqueeze(0) for path in sorted(glob(os.path.join(root, 'orthodontic/after/*.png')))]

for source in tqdm(sources):
    imgname = os.path.basename(source[1])
    source = source[0]
    with torch.no_grad():
        s = encoder.encode_image(preprocess(source))
    
    for i, (img1, img2) in (enumerate(zip(before_imgs, after_imgs))):
        gens = []
        
        with torch.no_grad():
            l1 = encoder.encode_image(preprocess(img1))
            l2 = encoder.encode_image(preprocess(img2))

        if mode == 'z_interpolate':
            latents = [alpha * (l2-l1) for alpha in alphas]
            
            for emb in latents:
                rows = []
                for class_idx in range(4):
                    # Labels. 
                    label = torch.zeros([1, G.c_dim], device=device)
                    label[:, class_idx] = 1
                    # z = torch.from_numpy(np.random.RandomState(seed).randn(1, emb.shape[1])).to(device)
                    # emb += z*0.1
                    # k = -2 * s@emb.T / (emb.norm()**2 + 1e-4)
                    # emb_tmp = s + emb*k

                    emb_tmp = s+emb
                    # emb_tmp = (emb_tmp - emb_tmp.mean()) / emb_tmp.std()

                    # emb_tmp = emb

                    # emb_tmp = torch.from_numpy(np.random.RandomState(seed).randn(1, emb.shape[1])).to(device)
                    # emb_tmp[0, idx[:10]] = -1
                    # emb_tmp[0, idx[10:]] = 3
                    img = G(label, img_emb=emb_tmp, noise_mode='random', truncation_psi=truncation_psi)
                    # img = G(z, label, img_emb=emb, noise_mode='const')
                    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
                    img = resizer(img).permute(1, 2, 0)
                    # img = (H, W, 3)
                    rows.append(img)
                img = torch.cat(rows, axis=0)
                gens.append(img)
            img = torch.cat(gens, axis=1)

        elif mode =='w_interpolate':
            for class_idx in range(4):
                cols = []
                # Labels. 
                label = torch.zeros([1, G.c_dim], device=device)
                label[:, class_idx] = 1
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
                w1 = G.mapping(label, img_emb=l1, truncation_psi=truncation_psi)
                w2 = G.mapping(label, img_emb=l2, truncation_psi=truncation_psi)
                latents = lerp(w1, w2, 5)
                for emb in latents:
                    img = G.synthesis(emb, noise_mode='const')
                    # img = G(z, label, img_emb=emb, noise_mode='const')
                    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
                    img = resizer(img).permute(1, 2, 0)
                    # img = (H, W, 3)
                    cols.append(img)
                img = torch.cat(cols, axis=1)
                gens.append(img)
            img = torch.cat(gens, axis=0)

        height = img.shape[0]

        img1 = source[0]

        img1 = resizer(img1)

        input_height = 180
        img1 = torch.nn.functional.pad(img1, (0, 10, int((height-input_height)/2), height - input_height -int((height-input_height)/2)), value=255).permute(1, 2, 0)

        img = torch.cat([img1, img], axis=1).to(dtype=torch.uint8)
        # imgname=f'k_times_{k}'
        Image.fromarray(img.cpu().numpy(), 'RGB').save(os.path.join(outdir, f'{imgname}_style_{i}_{mode}_trunc_{truncation_psi}.png'))
        # break