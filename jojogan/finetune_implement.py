"""
Modified version 
- from JoJoGAN / stylize.ipynb
- for AI server  inference


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
from model_minsu import *
from util import *
from e4e.models.psp import pSp
from e4e_projection import projection as e4e_projection

import argparse
#import wandb 


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


# TODO ------------------------------------------------------------
# [ ] Masking into latents
# [ ] Finetuning with masked latents
#   [X] novel perceptual loss : No need to do
#   [ ] arcface cosine similairy loss
# [ ] ReStyle
# [V] w+ style to s space
# [ ] s masking & finetuning


def finetune(args):
    print("-------------------- SETTING --------------------")
    print(f'LOG : input image : {args.img_path}')
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.skip_align == True : 
        aligned_face = Image.open(args.img_path).convert('RGB')     
    else:
        aligned_face = align_face(args.img_path)

    
    name = args.img_path.replace('.png', '_inversion.pt')
    inversion_model_path = f'{args.inversion_dir}/e4e_ffhq_encode.pt'
    ckpt = torch.load(inversion_model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = inversion_model_path
    opts= argparse.Namespace(**opts)
    
    inversion_net = pSp(opts, args.device).eval().to(args.device)
    my_w = e4e_projection( aligned_face, name, inversion_net, args.device).unsqueeze(0)

    # original generator 
    original_generator = Generator(1024, args.latent_dim, 8, 2).to(args.device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    
    original_generator.load_state_dict(ckpt["g_ema"], strict=False)
    mean_latent = original_generator.mean_latent(10000)
    generator = deepcopy(original_generator)
    mean_latent = original_generator.mean_latent(10000)
    
    # Debugging ------------------------------------------------------------
    # get s space test : dummy input 
    
    # s_space = generator.get_s_space(torch.zeros((1, 18, 512)).to(args.device))
    # dummy = generator.forward_with_s(s_space)

    # for idx, s in enumerate(s_space):
    #     print(f"style {idx}: {s.shape}")
    # print(dummy.shape)

    # load discriminator for perceptual loss
    discriminator = Discriminator(1024, 2).eval().to(args.device)
    ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
    discriminator.load_state_dict(ckpt["d"], strict=False)

    # reset generator
    del generator
    generator = deepcopy(original_generator)
    g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))


    # tranform 
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    style_path = f'./{args.input_dir}/{args.style_dir}'
    print(style_path)
    assert os.path.exists(style_path), f"INVALID INPUT : {style_path} does not exist!"

    style_name = args.style_dir.replace('/','_')
    print(f'LOG : style name : {style_name}')
    names = [ name.split('.')[0] for name in os.listdir(f'{style_path}') if name.endswith('png')or name.endswith('jpeg')or name.endswith('jpg')]
    targets = []
    latents = []
    
    os.makedirs(f'style_images_aligned/{args.style_dir}',exist_ok=True)
    os.makedirs(f'inversion_codes/{args.style_dir}',exist_ok=True)

    print("-------------------- STEP1 : GAN inversion --------------------")
    for name in names:
        # crop and align the face
        style_aligned_path = f'style_images_aligned/{args.style_dir}/{name}.png'
        
        if args.skip_align == True : 
            style_aligned = Image.open(f'./{args.input_dir}/{args.style_dir}/{name}.png').convert('RGB')
        else : 
            if not os.path.exists(style_aligned_path):
                style_aligned = align_face(f'{style_path}/{name}.png')
                style_aligned.save(style_aligned_path)
            else:
                style_aligned = Image.open(style_aligned_path).convert('RGB')
        
        # GAN invert
        style_code_path = f'inversion_codes/{args.style_dir}/{name}.pt'
        if not os.path.exists(style_code_path):
            latent = e4e_projection(style_aligned, style_code_path, inversion_net, args.device)
        else:
            latent = torch.load(style_code_path)['latent']

        targets.append(transform(style_aligned).to(args.device))
        latents.append(latent.to(args.device))

    targets = torch.stack(targets, 0)
    latents = torch.stack(latents, 0)


    target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))

    print("-------------------- STEP2 : Training set --------------------")

    batch = latents.shape[0]

    # NOTE #
    """
    [github]
    if args.preserve_color:
        id_swap = [9,11,15,16,17]
    else:
        id_swap = list(range(7, generator.n_latent))
    
    [ours]
    : Make random mask ({1,0}^26) (~= style mixing) and create training set
    - Toonify manner : don't mask coarse level (style idx 0 ~ 10). just mask fine level (style idx 11 ~ 26)
    - Implement eq.1 & eq.2 & other details described in paper
    - Note that random mask is static
    """
    # w+ space --> s space 
    # (B, 18, 512) --> [(B, 512), (B, 512), ... (B, 32)] << total 26 styles
    with torch.no_grad():
        s_space = generator.get_s_space(latents)
        mixing = generator.get_s_space(generator.get_latent(torch.randn([])))    

    style_mask = {}
    for level in range(args.level_mask):
        style_maks[level] = [random.sample(range(11, 27), level) for _ in range(args.mask_min_level, args.mask_max_level)]

    training_set_style_space = []
    for level, masks in style_mask.items():
        # for idx, s in enumerate(s_space):
        s_mixing = generator.get_s_space(
            generator.get_latent(torch.randn([batch, args.latent_dim]).to(args.device)))   # sFC(z_i) at eq2. s_i = M * s_i + (1-M) * s(FC(z_i)) 
        # TODO
        
        training_set_style_space.append(s)       
    

    for idx in tqdm(range(args.num_iter)):
        mean_w = generator.get_latent(torch.randn([batch, args.latent_dim]).to(args.device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        # FIXME : mean_w ????????? ?????? ??????? get_latent ??? ????????? mean_latent ??? ????????? ??? ??? ?????????...?

        in_latent = latents.clone()
        in_latent[:, id_swap] = args.alpha*latents[:, id_swap] + (1-args.alpha)*mean_w[:, id_swap]

        img = generator(in_latent, input_is_latent=True)
        
        with torch.no_grad():
            real_feat = discriminator(targets)

        fake_feat = discriminator(img)
        loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)
        g_optim.zero_grad()
        loss.backward()
        g_optim.step()

    #save stylegan_model
    stylegan_model_path = f'{args.stylegan_dir}/{style_name}.pt'
    torch.save({'g': generator.state_dict()},stylegan_model_path)
    print(f'LOG : StyleGAN model saved as {stylegan_model_path}!')

    ckpt = torch.load(stylegan_model_path, map_location=args.device)
    generator = Generator(1024, args.latent_dim, 8, 2).to(args.device)
    generator.load_state_dict(ckpt["g"], strict=False)


    print("-------------------- GENERATE RESULT --------------------")

    with torch.no_grad():
        generator.eval()
        z = torch.randn(args.n_sample, args.latent_dim, device=args.device)
        original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
        sample = generator([z], truncation=0.7, truncation_latent=mean_latent)
        original_my_sample = original_generator(my_w, input_is_latent=True)
        my_sample = generator(my_w, input_is_latent=True)

    face = transform(aligned_face).to(args.device).unsqueeze(0)

    my_output = torch.cat([face, my_sample], 0)
    img_name = (args.img_path).split('/')[-1][:-4]
    save_image(my_output,f'./{args.output_dir}/{style_name}_image_output.png')
    print(f'LOG : random output image saved as ./{args.output_dir}/{style_name}_image_output.png!')
    
    if args.random_output == True : 
        output = torch.cat([original_sample, sample], 0)
        save_image(output,f'./{args.output_dir}/{style_name}_random_output.png')
        print(f'LOG : output image saved as ./{args.output_dir}/{style_name}_random_output.png!')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,  default='dataset') 
    parser.add_argument('--style_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str,  default='output') 
    parser.add_argument('--img_path', type=str, default='test_input/iu.png')
    parser.add_argument('--alpha', type=float, default=0.0) 
    parser.add_argument('--preserve_color', type=bool, default = False)
    parser.add_argument('--num_mask', type=int, default=10)                     # ours
    parser.add_argument('--mask_max_level', type=int, default=1)                # ours
    parser.add_argument('--mask_min_level', type=int, default=0)                # ours
    parser.add_argument('--num_iter', type=int, default=300)                    # num of finetuning iteration
    parser.add_argument('--log_interval ', type=int, default=50)                # finetuning log interval
    parser.add_argument('--inversion_dir', type=str, default='models')
    parser.add_argument('--stylegan_dir', type=str, default='models')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_sample', type=int, default=5)
    parser.add_argument('--time_stamp', action="store_true")
    parser.add_argument('--device', type=str, default = "cuda:0")
    parser.add_argument('--latent_dim', type=int, default = 512)
    parser.add_argument('--random_output', type=bool, default = True)
    parser.add_argument('--skip_align', type=bool, default = False)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    finetune(args)





