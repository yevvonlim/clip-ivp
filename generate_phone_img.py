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

from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def _convert_image_to_rgb(image):
    return image.convert("RGB")



@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--condition', help='Conditional image path', metavar='PATH', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    condition:str,
    network_pkl: str,
    seeds: Optional[List[int]],
    noise_mode: str,
    outdir: str,
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    encoder, preprocess = clip.load(name='ViT-L/14', device=device)
    # preprocess = Compose([
    #     Resize(224, interpolation=BICUBIC),
    #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    # ])
    os.makedirs(outdir, exist_ok=True)
    condimg = PIL.Image.open(condition)
    desc = 'skewed to the left'
    text = clip.tokenize([desc]).to(device)
    
    imgToTensor = Compose([Resize((180, 300)),
                          _convert_image_to_rgb,
                            ToTensor(),])
    resizer = Resize((180, 300))
    rows = []
    for class_idx in range(4):
        # Labels. 
        label = torch.zeros([1, G.c_dim], device=device)
        label[:, class_idx] = 1
        with torch.no_grad():
            emb = encoder.encode_image(preprocess(condimg).to(device, dtype=torch.float32).unsqueeze(0))
            text_emb = encoder.encode_text(text)

        gen_imgs = []
        for seed in seeds:
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z=z, c=label, img_emb=text_emb, noise_mode=noise_mode)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
            img = resizer(img).permute(1, 2, 0)
            gen_imgs.append(img)

        # z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        # img = G(z=z, c=label, img_emb=text_emb, noise_mode=noise_mode)
        # img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
        # img = resizer(img).permute(1, 2, 0)
        # gen_imgs.append(img)

        # row = torch.cat([(imgToTensor(condimg) * 255).clamp(0, 255).to(device, dtype=torch.uint8).permute(1, 2, 0),
        #                 *gen_imgs], axis=1)
        row = torch.cat(gen_imgs, axis=1)
        rows.append(row)

    img = torch.cat(rows, axis=0)
    _id = os.path.basename(condition)

    _id = desc + '.png'
    PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(f'{outdir}/{_id}')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter