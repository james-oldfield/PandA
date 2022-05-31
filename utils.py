from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import torch
import numpy as np
from PIL import Image


def get_cols():
    # list of perceptually distinct colours (for spatial factor plots)
    return np.array([[255,0,0], [255,255,0], [0,234,255], [170,0,255], [255,127,0], [191,255,0], [0,149,255], [255,0,170], [255,212,0], [106,255,0], [0,64,255], [237,185,185], [185,215,237], [231,233,185], [220,185,237], [185,237,224], [143,35,35], [35,98,143], [143,106,35], [107,35,143], [79,143,35], [0,0,0], [115,115,115], [204,204,204]])


def mapRange(value, inMin, inMax, outMin, outMax):
    return outMin + (((value - inMin) / (inMax - inMin)) * (outMax - outMin))


def plot_masks(Us, r, s, rs=256, save_path=None, title_factors=True):
    """
    Plots the parts factors with matplotlib for visualization

    Parameters
    ----------
    Us : np.array
        Learnt parts factor matrix.
    r : int
        Number of factors to show.
    s : int
        Dimensions of each part (h*w).
    rs : int
        Target size to downsize images to.
    save_path : bool
        Save figure?
    title_factors : bool
        Print matplotlib title on each part?
    """

    fig = plt.figure(constrained_layout=True, figsize=(20, 3))
    spec = gridspec.GridSpec(ncols=r + 1, nrows=1, figure=fig)

    for i in range(0, r):
        fig.add_subplot(spec[i])

        if title_factors:
            plt.title(f'Part {i}')

        part = Us[i].reshape([s, s])
        part = mapRange(part, torch.min(part), torch.max(part), 0.0, 1.0) * 255
        part = part.detach().cpu().numpy()
        part = np.array(Image.fromarray(np.uint8(part)).convert('RGBA').resize((rs, rs), Image.NEAREST)) / 255

        plt.axis('off')
        plt.imshow(part, vmin=1, vmax=1, cmap='gray', alpha=1.00)

    if save_path is not None:
        plt.savefig(save_path)


def plot_colours(image, Us, r, s, rs=128, save_path=None, alpha=1.0, seed=-1, legend=True):
    """
    Plots the parts factors over an image with matplotlib for visualization

    Parameters
    ----------
    image : np.array
        Image to visualize.
    Us : np.array
        Learnt parts factor matrix.
    r : int
        Number of factors to show.
    s : int
        Dimensions of each part (h*w).
    rs : int
        Target size to downsize images to.
    alpha : float
        Alpha value for the masks.
    seed : int
        Random seed when generating the colour palette (use -1 to use the provided "perceptually distinct" colour palette, but note this has a maximum of 30 colours or so).
    legend : bool
        Plot the legend, detailing the colour-coded parts key?
    """

    img = Image.fromarray(image).resize((rs, rs)).convert('RGBA')

    # Use perceptually distinct colour list, or random seed (for e.g. if you have too many factors)
    cols = get_cols()
    if seed >= 0:
        np.random.seed(seed)
        cols = np.random.randint(0, 255, [r, 3])

    plt.imshow(img, alpha=1.0)
    plt.axis('off')

    patches = []
    for i in range(0, r):
        mask = Us[i].detach().cpu().numpy().reshape([s, s])
        mask = mapRange(mask, np.min(mask), np.max(mask), 0, 255)
        mask = np.uint8(mask)
        mask = np.array(Image.fromarray(mask).convert('L').resize((rs, rs)))
        mask = (mask[:, :, None] / 255.) * np.array(np.concatenate([cols[i] / 255, [1]]))

        patches += [mpatches.Patch(color=cols[i] / 255, label=f'Part {i}')]

        plt.imshow(mask, vmin=0, vmax=1, alpha=alpha)

    if legend:
        plt.legend(title='Spatial factors', handles=patches, bbox_to_anchor=(1.01, 1.01), loc="upper left")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)