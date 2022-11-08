import torch
from torchvision import transforms, utils
from backup.util import *
import os
from yaiverse.inference.importFiles.model import *
from yaiverse.inference.importFiles.e4e_projection import projection as e4e_projection
from yaiverse.inference.importFiles.psp import pSp
from backup.util import *
from yaiverse.inference.mediapipe import *
import argparse

class ConvertModel:
    
    """
    Load static models to memory
    """
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
        
        # Inference dummy data
        self.generate_face("dummy", "sketch_multi")
        
        
    """
    Inference input image with style and save image
    """
    def generate_face(self, col:str, style:str) -> None:
        stylegan_model_path = self.dir + '/yaiverse/inference/models/style_model/{}.pt'.format(style)
        ckpt = torch.load(stylegan_model_path, map_location=self.device)

        generator = Generator(1024, self.latent_dim, 8, 2).to(self.device)
        generator.load_state_dict(ckpt["g"], strict=False)
        input_img_path = os.path.join(self.dir + "data", col + '/image.jpg')
        my_w = self.align_inversion(input_img_path)
        
        noise = generator.make_noise()
        my_toonify = generator(my_w, truncation=0.7,input_is_latent=True, noise=noise)
        transform = transforms.ToPILImage()
        my_toonify = utils.make_grid(my_toonify, normalize=True, range=(-1, 1)).squeeze(0)
        my_toonify = transform(my_toonify)
        output_path = os.path.join(self.dir + "data", col + '/result.jpg')
        my_toonify.save(output_path)
        
        
    """
    Align & crop image
    return inversion vector
    """
    def align_inversion(self, img_path:str):
        aligned_face = face_detection(img_path)
        return e4e_projection(
                            aligned_face,
                            name=img_path.replace('.jpg', '_inversion.pt'),
                            net=self.inversion_net,
                            save_inverted=False,
                            device=self.device).unsqueeze(0)
        
    def get_latent(self):

        transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        names = os.listdir("/home/yai/faceAlign/i_data")

        targets = []
        latents = []

        for name in names:
            style_path = os.path.join('/home/yai/faceAlign/i_data', name)
            assert os.path.exists(style_path), f"{style_path} does not exist!"

            name = strip_path_extension(name)

            # crop and align the face
            style_aligned_path = os.path.join('/home/yai/faceAlign/i_data', f'{name}.png')
            print(name)
            #if not os.path.exists(style_aligned_path):
            # style_aligned = align_face(style_path)
            #style_aligned = Image.open(style_path)
            # style_aligned.save(style_aligned_path)
            #else:
            style_aligned = Image.open(style_aligned_path).convert('RGB')

            # GAN invert
            style_code_path = os.path.join('/home/yai/backend/yaiverse/inference/images/inversion_codes', f'{name}.pt')
            if not os.path.exists(style_code_path):
                latent = e4e_projection(style_aligned, style_code_path,net=self.inversion_net, device = self.device)
            else:
                latent = torch.load(style_code_path)['latent']

            targets.append(transform(style_aligned).to(self.device))
            latents.append(latent.to(self.device))

        targets = torch.stack(targets, 0)
        latents = torch.stack(latents, 0)
        torch.save(latent, "/home/yai/backend/yaiverse/inference/images/inversion_codes/latent.pt")
        return latents, targets
        
        
    def test(self, col, style, randnum):
        stylegan_model_path = self.dir + '/yaiverse/inference/models/style_model/{}.pt'.format(style)
        # stylegan_model_path = '/home/yai/backend/yaiverse/inference/images/inversion_codes/latent.pt'
        ckpt = torch.load(stylegan_model_path, map_location=self.device)

        generator = Generator(1024, self.latent_dim, 8, 2).to(self.device)
        generator.load_state_dict(ckpt["g"], strict=False)
        input_img_path = os.path.join(self.dir + "data", col + '/image.jpg')
        my_w = self.align_inversion(input_img_path)
        
        noise = generator.make_noise()
        # latent = torch.load("/home/yai/backend/yaiverse/inference/images/inversion_codes/latent.pt")
        # # rand_num =random.randint(0,len(latent)-1)
        # # print(rand_num)
        # trun = latent[randnum].unsqueeze(0)
        # trun =  (torch.randn(*trun.size())*2).to(self.device)
        # my_toonify = generator(my_w, truncation=0.7, truncation_latent= trun, noise=noise, input_is_latent=True)
        my_toonify = generator(my_w, truncation=0.7,input_is_latent=True, noise=noise)
        transform = transforms.ToPILImage()
        my_toonify = utils.make_grid(my_toonify, normalize=True, range=(-1, 1)).squeeze(0)
        my_toonify = transform(my_toonify)
        output_path = os.path.join(self.dir + "data", col + '/result.jpg')
        my_toonify.save(output_path)
        
            
