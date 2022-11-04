import torch
# torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from yaiverse.inference.importFiles.util import *
import os
from copy import deepcopy
from yaiverse.inference.importFiles.model import *
from yaiverse.inference.importFiles.e4e_projection import projection as e4e_projection
from yaiverse.inference.importFiles.psp import pSp
from yaiverse.inference.importFiles.util import *
import argparse

class ConvertModel:
    def __init__(self):
        self.dir = "/home/yai/backend/"
        self.device = "cuda:0"
        self.latent_dim = 512
        inversion_model_path = self.dir + 'yaiverse/inference/importFiles/e4e_ffhq_encode.pt'
        ckpt = torch.load(inversion_model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = inversion_model_path
        opts= argparse.Namespace(**opts)
        self.inversion_net = pSp(opts, self.device).eval().to(self.device)
        
        
        
    def generate_face(self, col, style):
        stylegan_model_path = self.dir + '/yaiverse/inference/models/style_model/{}.pt'.format(style)
        ckpt = torch.load(stylegan_model_path, map_location=self.device)

        generator = Generator(1024, self.latent_dim, 8, 2).to(self.device)
        generator.load_state_dict(ckpt["g"], strict=False)
        input_img_path = os.path.join(self.dir + "data", col + '/image.jpg')
        my_w = self.align_inversion(input_img_path)
        my_toonify = generator(my_w, input_is_latent=True)
        transform = transforms.ToPILImage()
        my_toonify = utils.make_grid(my_toonify, normalize=True, range=(-1, 1)).squeeze(0)
        my_toonify = transform(my_toonify)
        output_path = os.path.join(self.dir + "data", col + '/result.jpg')
        my_toonify.save(output_path)
        
        

    
    def align_inversion(self, img_path):
        aligned_face = align_face(img_path)
        return e4e_projection(
                            aligned_face,
                            name=img_path.replace('.png', '_inversion.pt'),
                            net=self.inversion_net,
                            save_inverted=False,
                            device=self.device).unsqueeze(0)
        
        
        
        