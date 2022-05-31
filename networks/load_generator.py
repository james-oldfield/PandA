from networks.genforce.models import MODEL_ZOO
from networks.genforce.models import build_generator
from networks.biggan import BigGAN
from networks.stylegan3.load_stylegan3 import load_stylegan3
import os
import subprocess
import torch


def load_generator(model_name, device, CHECKPOINT_DIR='./models'):

    print(f'Building generator for model `{model_name}` ...')

    if 'stylegan3' in model_name:
        generator = load_stylegan3(model_name, device)
    elif model_name == 'biggan':
        generator = BigGAN.from_pretrained('biggan-deep-256')
        generator.z_space_dim = 128
    else:
        model_config = MODEL_ZOO[model_name].copy()
        url = model_config.pop('url')  # URL to download model if needed.

        generator = build_generator(**model_config)
        print(f'Finish building generator.')

        # Load pre-trained weights.
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name + '.pth')
        # print(f'Loading checkpoint from `{checkpoint_path}` ...')
        print('Loading checkpoint')
        if not os.path.exists(checkpoint_path):
            print(f'  Downloading checkpoint from `{url}` ...')
            subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
            print(f'  Finish downloading checkpoint.')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'generator_smooth' in checkpoint:
            generator.load_state_dict(checkpoint['generator_smooth'])
        else:
            generator.load_state_dict(checkpoint['generator'])
        print(f'Finish loading checkpoint.')

    generator.eval()
    generator.to(device)
    return generator