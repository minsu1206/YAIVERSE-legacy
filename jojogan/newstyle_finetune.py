"""
Modified version 
- from JoJoGAN / stylize.ipynb
- for Pretraining and Finetuning


"""
from copy import deepcopy
import time
import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from torchvision.utils import save_image
from PIL import Image
import math
import random
import os
import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
from model import *
from util import *
from e4e.models.psp import pSp
from e4e_projection import projection as e4e_projection

import argparse
import wandb 


def time_stamp(func_name, start, template1='TIME', return_val=False):
    stamp = round(time.time() - start, 5)
    print(f'{template1} : {func_name} : {stamp} s')
    if return_val:
        return stamp


def time_stamp_val(func_name, elapse, template1='TIME', return_val=False):
    elapse = round(elapse, 5)
    print(f'{template1} : {func_name} : {elapse} s')
    if return_val:
        return elapse



def finetune(args):

    print("-------------------- INPUT IMG --------------------")
    print(f'## input image : {args.img_path}')
    aligned_face = align_face(args.img_path)
    name = args.img_path.replace('.png', '_inversion.pt')
    my_w = e4e_projection(aligned_face, name, args.device).unsqueeze(0)

    #original generator 
    original_generator = Generator(1024, args.latent_dim, 8, 2).to(args.device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    mean_latent = original_generator.mean_latent(10000)
    generator = deepcopy(original_generator)
    mean_latent = original_generator.mean_latent(10000)

    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    print("-------------------- STYLE IMG --------------------")
    style_path = f'./{args.input_dir}/{args.style_dir}'
    assert os.path.exists(style_path), f"{style_path} does not exist!"
    style_name = args.style_dir.replace('/','_')
    print(f'## style name : {style_name}')
    names = [ name.split('.')[0] for name in os.listdir(f'{style_path}') if name.endswith('png')or name.endswith('jpeg')or name.endswith('jpg')]
    targets = []
    latents = []


    os.makedirs(f'style_images_aligned/{args.style_dir}',exist_ok=True)
    os.makedirs(f'inversion_codes/{args.style_dir}',exist_ok=True)

    for name in names:
       
        # crop and align the face
        style_aligned_path = f'style_images_aligned/{args.style_dir}/{name}.png'
        if not os.path.exists(style_aligned_path):
            style_aligned = align_face(f'{style_path}/{name}.png' )
            style_aligned.save(style_aligned_path)
        else:
            style_aligned = Image.open(style_aligned_path).convert('RGB')

        # GAN invert
        style_code_path = f'inversion_codes/{args.style_dir}/{name}.pt'
        if not os.path.exists(style_code_path):
            latent = e4e_projection(style_aligned, style_code_path, args.device)
        else:
            latent = torch.load(style_code_path)['latent']

        targets.append(transform(style_aligned).to(args.device))
        latents.append(latent.to(args.device))

    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)

    target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))
    #display_image(target_im, title='Style References')

    print("-------------------- FINETUNE StyleGAN --------------------")

    if args.use_wandb:
        wandb.init(project="JoJoGAN")
        config = wandb.config
        config.num_iter = args.num_iter
        config.preserve_color = args.preserve_color
        wandb.log(
        {"Style reference": [wandb.Image(transforms.ToPILImage()(target_im))]},
        step=0)

    # load discriminator for perceptual loss
    discriminator = Discriminator(1024, 2).eval().to(args.device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    discriminator.load_state_dict(ckpt["d"], strict=False)

    # reset generator
    del generator
    generator = deepcopy(original_generator)
    g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

    # Which layers to swap for generating a family of plausible real images -> fake image
    if args.preserve_color:
        id_swap = [9,11,15,16,17]
    else:
        id_swap = list(range(7, generator.n_latent))

    for idx in tqdm(range(args.num_iter)):
        mean_w = generator.get_latent(torch.randn([latents.size(0), args.latent_dim]).to(args.device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        in_latent = latents.clone()
        in_latent[:, id_swap] = args.alpha*latents[:, id_swap] + (1-args.alpha)*mean_w[:, id_swap]

        img = generator(in_latent, input_is_latent=True)

        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(img)

        loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)
    
        if args.use_wandb:
            wandb.log({"loss": loss}, step=idx)
            if idx % args.log_interval == 0:
                generator.eval()
                my_sample = generator(my_w, input_is_latent=True)
                generator.train()
                my_sample = transforms.ToPILImage()(utils.make_grid(my_sample, normalize=True, range=(-1, 1)))
                wandb.log(
                {"Current stylization": [wandb.Image(my_sample)]},
                step=idx)

        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    #save stylegan_model
    stylegan_model_path = f'{args.stylegan_dir}/{style_name}.pt'
    torch.save({'g': generator.state_dict()},stylegan_model_path)
    print(f'## StyleGAN model saved as {stylegan_model_path}!')

    ckpt = torch.load(stylegan_model_path, map_location=args.device)
    generator = Generator(1024, args.latent_dim, 8, 2).to(args.device)
    generator.load_state_dict(ckpt["g"], strict=False)


    print("-------------------- GENERATE RESULT --------------------")
    torch.manual_seed(args.seed)

    with torch.no_grad():
        generator.eval()
        z = torch.randn(args.n_sample, args.latent_dim, device=args.device)
        original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
        sample = generator([z], truncation=0.7, truncation_latent=mean_latent)
        original_my_sample = original_generator(my_w, input_is_latent=True)
        my_sample = generator(my_w, input_is_latent=True)

    # display reference images
    style_images = []
    for name in names:
        style_image = transform(Image.open(f'{style_path}/{name}.png'))
        style_images.append(style_image)
        
    face = transform(aligned_face).to(args.device).unsqueeze(0)
    # style_images = torch.stack(style_images, 0).to(args.device)
    my_output = torch.cat([face, my_sample], 0)
    output = torch.cat([original_sample, sample], 0)

    # save_image(style_image,f'./{args.output_dir}/{style_name}_refernce.png')
    save_image(my_output,f'./{args.output_dir}/{style_name}_image_output.png')
    save_image(output,f'./{args.output_dir}/{style_name}_random_output.png')

    # print(f'## reference image saved as ./{args.output_dir}/{style_name}_refernce.png!')
    print(f'## random output image saved as ./{args.output_dir}/{style_name}_image_output.png!')
    print(f'## output image saved as ./{args.output_dir}/{style_name}_random_output.png!')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,  required=True) 
    parser.add_argument('--style_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--img_path', type=str, default='./test_input/image.png')
    parser.add_argument('--alpha', type=float, default=0.0) 
    parser.add_argument('--preserve_color', type=bool, default = True)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--log_interval ', type=int, default=50)
    parser.add_argument('--inversion_dir', type=str, default='models')
    parser.add_argument('--stylegan_dir', type=str, default='models')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_sample', type=int, default=5)

    parser.add_argument('--time_stamp', action="store_true")
    parser.add_argument('--device', type=str, default = "cuda")
    parser.add_argument('--latent_dim', type=int, default = 512)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    finetune(args)




