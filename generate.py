import argparse

import torch
from model import Generator
from tqdm import tqdm

from PIL import Image
import numpy as np

import random

import os

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


def save_gif(img_name, img_ar):

    img_ar = make_image(img_ar)

    pil_img = Image.fromarray(img_ar[0])
    pil_img.save(img_name, 
                 save_all=True, 
                 append_images = [Image.fromarray(img_ar[i]) for i in range(1,len(img_ar))], 
                 duration=100, 
                 loop=0)


def save_png(img_name, img_ar):

    img_ar = make_image(img_ar)

    pil_img = Image.fromarray(img_ar[0])
    pil_img.save(img_name)



def wrapped_generate(pics, ckpt, device, mean_latent, output_dir='output'):
    g_ema = Generator(
    128, 512, 8, channel_multiplier=2
    ).to(device)
    checkpoint = torch.load(ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])
    return generate(pics, g_ema, device, mean_latent, output_dir)

def wrapped_generate2(pics, ckpt, device, mean_latent, output_dir='output'):
    g_ema = Generator(
    128, 512, 8, channel_multiplier=2
    ).to(device)
    checkpoint = torch.load(ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])
    return generate2(pics, g_ema, device, mean_latent, output_dir)

def wrapped_generate2p(ckpt, device, mean_latent, center_latent, r1, mj, filename):
    g_ema = Generator(
    128, 512, 8, channel_multiplier=2
    ).to(device)
    checkpoint = torch.load(ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])
    return generate2p(g_ema, device, mean_latent, center_latent, r1, mj, filename)


def generate(pics, g_ema, device, mean_latent, output_dir='output'):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(pics)):
            sample_z = torch.randn(1, 512, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=1, truncation_latent=mean_latent
            )

            save_png(f"{output_dir}/{str(i).zfill(6)}.png", sample)


def generate2(pics, g_ema, device, mean_latent, output_dir='output'):

    mj = 50
    r1 = 0.25

    with torch.no_grad():
        g_ema.eval()
        sample_z0 = torch.randn(512, device=device)
        for i in tqdm(range(pics)):
            sample_z1 = torch.randn(512, device=device)
            axis = random.sample(range(512), 2)
            sample_tmp = []
            for j in range(mj):
                tmp = torch.zeros(512, device=device)
                tmp[0::2] = np.cos(j * np.pi * 2 / mj)
                tmp[1::2] = np.sin(j * np.pi * 2 / mj)

                sample_tmp.append(sample_z0 + r1 * tmp)

            sample_z = torch.stack(sample_tmp)

            sample, _ = g_ema(
                [sample_z], truncation=1, truncation_latent=mean_latent
            )

            save_gif(f"{output_dir}/{str(i).zfill(6)}.gif", sample)

            sample_z0 = sample_z1

def generate2p(g_ema, device, mean_latent, center_latent, r1, mj, filename):

    with torch.no_grad():
        g_ema.eval()
        sample_z0 = center_latent

        sample_tmp = [sample_z0]
        for j in range(mj):
            tmp = torch.zeros(512, device=device)
            tmp[0::2] = np.cos(j * np.pi * 2 / mj)
            tmp[1::2] = np.sin(j * np.pi * 2 / mj)

            sample_tmp.append(sample_z0 + r1 * tmp)

        sample_z = torch.stack(sample_tmp)

        sample, _ = g_ema(
            [sample_z], truncation=1, truncation_latent=mean_latent
        )

        save_gif(filename, sample[1:])
        save_png(filename.replace('gif', 'png'), sample)




if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")
    parser.add_argument(
        "--outputdir", default='sample', help="output image folder"
    )

    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )

    args = parser.parse_args()

    os.makedirs(args.outputdir, exist_ok=True)



    mean_latent = None

    for i in range(args.pics):
        center_latent = torch.randn(512, device=device)
        generate2p(args, g_ema, device, mean_latent, center_latent, 0.25, 16, f"{args.outputdir}/{str(i).zfill(6)}.gif")

    
