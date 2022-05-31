import sys
sys.path.append('./networks/stylegan3/')
from torch_utils import misc
import numpy as np
import legacy
import dnnlib
from PIL import Image

def make_transform(translate, angle):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


def load_stylegan3(gan_type, device):
    # load models
    network_pkl = f'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/{gan_type}.pkl'
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        # resume_data = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        resume_data = legacy.load_network_pkl(f)

    invar_type = str(network_pkl.split('/')[-1].split('-')[1])
    res = int(network_pkl.split('.')[-2].split('x')[-1])
    batch_size = 1

    # init blank model
    G_kwargs = dnnlib.EasyDict(
        class_name='training.networks_stylegan3.Generator',
        magnitude_ema_beta=0.5 ** (batch_size / (20 * 1e3)),
        z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(),
        channel_base=32768,
        channel_max=512,
    )

    if invar_type == 'r':
        G_kwargs.conv_kernel = 1 # Use 1x1 convolutions.
        G_kwargs.channel_base *= 2 # Double the number of feature maps.
        G_kwargs.channel_max *= 2
        G_kwargs.use_radial_filters = True # Use radially symmetric downsampling filters.

    G_kwargs.mapping_kwargs.num_layers = 2 

    common_kwargs = dict(c_dim=0, img_resolution=res, img_channels=3)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G.z_space_dim = G.z_dim

    # copy params to blank model
    for name, module in [('G_ema', G)]:
        misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    del resume_data

    return G
