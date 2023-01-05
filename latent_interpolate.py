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


class CLIP_Projector(torch.nn.Module):
    def __init__(self, encoder, projector, device):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.device = device

    def forward(self, x):
        with torch.no_grad():
            x.to(self.device)
            return self.projector(self.encoder.encode_image(x))

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
network_pkl = '/root/stylegan2-ada-intraoral/class-conditional/cond-auto1-gamma5-resumecustom/network-snapshot-012000.pkl'
print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

encoder, _ = clip.load(name='ViT-L/14', device=device)
preprocess = Compose([
    Resize(encoder.visual.input_resolution, interpolation=BICUBIC),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

root = '/root/stylegan2-ada-intraoral/latent_projection/'
outdir = root
resizer = Resize((180, 300))
seeds = [0, 1, 11, 5, 97]

mode = 'z_interpolate'
truncation_psi = 0.7
alpha = 1
start_imgs = [(torch.from_numpy(np.array(Image.open(path)).transpose(2, 0, 1)).to(device, dtype=torch.float32).unsqueeze(0), path) for path in sorted(glob(os.path.join(root, 'data/start/*.png')))]
end_imgs = [torch.from_numpy(np.array(Image.open(path)).transpose(2, 0, 1)).to(device, dtype=torch.float32).unsqueeze(0) for path in sorted(glob(os.path.join(root, 'data/end/*.png')))]

for seed in tqdm(seeds):
    for i, (img1, img2) in (enumerate(zip(start_imgs, end_imgs))):
        gens = []
        imgname = os.path.basename(img1[1])
        img1 = img1[0]
        with torch.no_grad():
            l1 = encoder.encode_image(preprocess(img1))
            l2 = encoder.encode_image(preprocess(img2))

        if mode == 'z_interpolate':
            latents = lerp(l1, l2, 5)

            for emb in latents:
                rows = []
                for class_idx in range(4):
                    # Labels. 
                    label = torch.zeros([1, G.c_dim], device=device)
                    label[:, class_idx] = 1
                    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
                    img = G(z, label, img_emb=emb, noise_mode='random', truncation_psi=truncation_psi)
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
                w1 = G.mapping(z, label, img_emb=l1, truncation_psi=truncation_psi)
                w2 = G.mapping(z, label, img_emb=l2, truncation_psi=truncation_psi)
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

        img1 = img1[0]
        img1 = resizer(img1)

        input_height = 180
        img1 = torch.nn.functional.pad(img1, (0, 10, int((height-input_height)/2), height - input_height -int((height-input_height)/2)), value=255).permute(1, 2, 0)

        img = torch.cat([img1, img], axis=1)
        img2 = img2[0]
        img2 = resizer(img2)
        img2 = torch.nn.functional.pad(img2, (10, 0, int((height-input_height)/2), height - input_height - int((height-input_height)/2)), value=255).permute(1, 2, 0)

        img = torch.cat([img, img2], axis=1).to(dtype=torch.uint8)
        Image.fromarray(img.cpu().numpy(), 'RGB').save(os.path.join(outdir, f'{imgname}_seed{seed}_{mode}_trunc_{truncation_psi}.png'))
