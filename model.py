from networks.load_generator import load_generator
from networks.genforce.utils.visualizer import postprocess_image as postprocess
from networks.biggan import one_hot_from_names, truncated_noise_sample
from networks.stylegan3.load_stylegan3 import make_transform

from matplotlib import pyplot as plt

from utils import plot_masks, plot_colours, mapRange

import torch
import numpy as np
from PIL import Image
import tensorly as tl
tl.set_backend('pytorch')


class Model():
    def __init__(self, model_name, t=0, layer=5, trunc_psi=1.0, trunc_layers=18, device='cuda', biggan_classes=['fox']):
        """
        Instantiate the model for decomposition and/or local image editing.

        Parameters
        ----------
        model_name : string
            Name of architecture and dataset--one of the items in ./networks/genforce/models/model_zoo.py.
        t : int
            Random seed for the generator (to generate a sample image).
        layer : int
            Intermediate layer at which to perform the decomposition.
        trunc_psi : float
            Truncation value in [0, 1].
        trunc_layers : int
            Number of layers at which to apply truncation.
        device : string
            Device to store the tensors on.
        biggan_classes : list
            List of strings specifying imagenet classes of interest (e.g. ['alp', 'breakwater']).
        """
        self.gan_type = model_name.split('_')[0]
        self.model_name = model_name
        self.randomize_noise = False
        self.device = device
        self.biggan_classes = biggan_classes
        self.layer = layer  # layer to decompose
        self.start = 0 if 'stylegan2' in self.gan_type else 2
        self.trunc_psi = trunc_psi
        self.trunc_layers = trunc_layers

        self.generator = load_generator(model_name, device)
        noise = torch.Tensor(np.random.randn(1, self.generator.z_space_dim)).to(self.device)
        z, image = self.sample(noise, layer=layer, trunc_psi=trunc_psi, trunc_layers=trunc_layers, verbose=True)

        self.c = z.shape[1]
        self.s = z.shape[2]
        self.image = image

    def HOSVD(self, batch_size=10, n_iters=100):
        """
        Initialises the appearance basis A. In particular, computes the left-singular vectors of the channel mode's scatter matrix.

        Note: total samples used is batch_size * n_iters

        Parameters
        ----------
        batch_size : int
            Number of activations to sample in a single go.
        n_iters : int
            Number of times to sample `batch_size`-many activations.
        """
        np.random.seed(0)
        torch.manual_seed(0)

        with torch.no_grad():
            Z = torch.zeros((batch_size * n_iters, self.c, self.s, self.s), device=self.device)

            # note: perform in loops to have a larger effective batch size
            print('Starting loops...')
            for i in range(n_iters):
                np.random.seed(i)
                torch.manual_seed(i)
                noise = torch.Tensor(np.random.randn(batch_size, self.generator.z_space_dim)).to(self.device)
                z, _ = self.sample(noise, layer=self.layer, partial=True)

                Z[(batch_size * i):(batch_size * (i + 1))] = z

            Z = Z.view([-1, self.c, self.s**2])
            print(f'Generated {batch_size * n_iters} gan samples...')

            scat = 0
            for _, x in enumerate(Z):
                # mode-3 unfolding in the paper, but in PyTorch channel mode is first.
                m_unfold = tl.unfold(x, 0)
                scat += m_unfold @ m_unfold.T

            self.Uc_init, _, _ = np.linalg.svd((scat / len(Z)).cpu().numpy())
            self.Uc_init = torch.Tensor(self.Uc_init).to(self.device)

            print('... HOSVD done')

    def decompose(self, ranks=[512, 8], lr=1e-8, batch_size=1, its=10000, log_modulo=1000, hosvd_init=True, stochastic=True, n_iters=1, verbose=True):
        """
        Performs the decomposition in the paper. In particular, Algorithm 1.,
        either with a non-fixed batch of samples (stochastic=True), or descends the full gradients.

        Parameters
        ----------
        ranks : list
            List of integers specifying the R_C and R_S, the ranks--i.e. number of parts and appearances respectively.
        lr : float
            Learning rate the projected gradient descent.
        batch_size : int
            Number of samples in each batch.
        its : int
            Total number of iterations.
        log_modulo : int
            Parameter used to control how often "training" information is displayed.
        hosvd_init : bool
            Initialise appearance factors from HOSVD? (else from random normal).
        stochastic : bool
            Sample the batch again each iteration? Else descent full gradients
        n_iters : int
            Number of `batch_size`-many samples to take (for full gradient).
            The total activations are sampled in batches in a loop to enable it to fit in memory.
        verbose : bool
            Prints extra information.
        """
        self.ranks = ranks
        np.random.seed(0)
        torch.manual_seed(0)

        #######################
        # init from HOSVD, else random normal
        Uc = self.Uc_init[:, :ranks[0]].detach().clone().to(self.device) if hosvd_init else torch.randn(self.Uc_init.shape[0], ranks[0]).detach().clone().to(self.device) * 0.01
        Us = torch.Tensor(np.random.uniform(0, 0.01, size=[self.s**2, ranks[1]])).to(self.device)
        #######################

        print(f'Uc shape: {Uc.shape}, Us shape: {Us.shape}')

        with torch.no_grad():
            zeros = torch.zeros_like(Us, device=self.device)
            Us = torch.maximum(Us, zeros)

            # use a fixed batch (i.e. descend the full gradient)
            if not stochastic:
                Z = torch.zeros((batch_size * n_iters, self.c, self.s, self.s), device=self.device)

                # note: perform in loops to have a larger effective batch size
                print(f'Starting loops, total Z shape: {Z.shape}...')
                for i in range(n_iters):
                    np.random.seed(i)
                    torch.manual_seed(i)
                    noise = torch.Tensor(np.random.randn(batch_size, self.generator.z_space_dim)).to(self.device)
                    z, _ = self.sample(noise, layer=self.layer, partial=True)

                    Z[(batch_size * i):(batch_size * (i + 1))] = z

            for t in range(its):
                np.random.seed(t)
                torch.manual_seed(t)

                # resample the batch, if stochastic
                if stochastic:
                    noise = torch.Tensor(np.random.randn(batch_size, self.generator.z_space_dim)).to(self.device)
                    Z, _ = self.sample(noise, layer=self.layer, partial=True)

                if verbose:
                    # reconstruct (for visualisation)
                    coords = tl.tenalg.multi_mode_dot(Z.view(-1, self.c, self.s**2).float(), [Uc.T, Us.T], transpose=False, modes=[1, 2])
                    Z_rec = tl.tenalg.multi_mode_dot(coords, [Uc, Us], transpose=False, modes=[1, 2])

                    self.rec_loss = torch.mean(torch.norm(Z.view(-1, self.c, self.s**2).float() - Z_rec, p='fro', dim=[1, 2]) ** 2)

                # Update S
                z = Z.view(-1, self.c, self.s**2).float()
                Us_g = -4 * (torch.transpose(z,1,2)@Uc@Uc.T@z@Us) + \
                    2 * (Us@Us.T@torch.transpose(z,1,2)@Uc@Uc.T@Uc@Uc.T@z@Us + torch.transpose(z,1,2)@Uc@Uc.T@Uc@Uc.T@z@Us@Us.T@Us)
                Us_g = torch.sum(Us_g, 0)

                Us = Us - lr * Us_g
                # --- projection step ---a
                Us = torch.maximum(Us, zeros)

                # Update C
                Uc_g = -4 * (z@Us@Us.T@torch.transpose(z,1,2)@Uc) + \
                    2 * (Uc@Uc.T@z@Us@Us.T@Us@Us.T@torch.transpose(z,1,2)@Uc + z@Us@Us.T@Us@Us.T@torch.transpose(z,1,2)@Uc@Uc.T@Uc)
                Uc_g = torch.sum(Uc_g, 0)
                Uc = Uc - lr * Uc_g

                self.Us = Us
                self.Uc = Uc

                if t % log_modulo == 0 and verbose:
                    print(f'ITERATION: {t}')
                    z, x = self.sample(noise, layer=self.layer, partial=False)

                    # here we display the learnt parts factors and also overlay them over the images to visualise.
                    plot_masks(Us.T, r=min(ranks[-1], 32), s=self.s)
                    plt.show()
                    plot_colours(x, Us.T, r=ranks[-1], s=self.s, seed=-1)
                    plt.show()

    def decompose_autograd(self, ranks=[512, 8], lr=1e-8, batch_size=1, its=10000, log_modulo=1000, verbose=True, hosvd_init=True):
        """
        Performs the same decomposition in the paper, only uses autograd with Adam optimizer (and projected gradient descent).

        Parameters
        ----------
        ranks : list
            List of integers specifying the R_C and R_S, the ranks--i.e. number of parts and appearances respectively.
        lr : float
            Learning rate the projected gradient descent.
        batch_size : int
            Number of samples in each batch.
        its : int
            Total number of iterations.
        log_modulo : int
            Parameter used to control how often "training" information is displayed.
        hosvd_init : bool
            Initialise appearance factors from HOSVD? (else from random normal).
        verbose : bool
            Prints extra information.
        """
        self.ranks = ranks
        np.random.seed(0)
        torch.manual_seed(0)

        #######################
        # init from HOSVD, else random normal
        Uc = torch.nn.Parameter(self.Uc_init[:, :ranks[0]].detach().clone().to(self.device), requires_grad=True) \
            if hosvd_init else torch.nn.Parameter(torch.randn(self.Uc_init.shape[0], ranks[0]).detach().clone().to(self.device) * 0.01)
        Us = torch.nn.Parameter(torch.Tensor(np.random.uniform(0, 0.01, size=[self.s**2, ranks[1]])).to(self.device), requires_grad=True)
        #######################
        optimizerS = torch.optim.Adam([Us], lr=lr)
        optimizerC = torch.optim.Adam([Uc], lr=lr)

        print(f'Uc shape: {Uc.shape}, Us shape: {Us.shape}')

        zeros = torch.zeros_like(Us, device=self.device)
        for t in range(its):
            np.random.seed(t)
            torch.manual_seed(t)

            noise = torch.Tensor(np.random.randn(batch_size, self.generator.z_space_dim)).to(self.device)
            Z, _ = self.sample(noise, layer=self.layer, partial=True)

            # Update S
            # reconstruct
            coords = tl.tenalg.multi_mode_dot(Z.view(-1, self.c, self.s**2).float(), [Uc.T, Us.T], transpose=False, modes=[1, 2])
            Z_rec = tl.tenalg.multi_mode_dot(coords, [Uc, Us], transpose=False, modes=[1, 2])

            rec_loss = torch.mean(torch.norm(Z.view(-1, self.c, self.s**2).float() - Z_rec, p='fro', dim=[1, 2]) ** 2)
            rec_loss.backward(retain_graph=True)

            optimizerS.step()
            # --- projection step ---
            Us.data = torch.maximum(Us.data, zeros)
            optimizerS.zero_grad()
            optimizerC.zero_grad()

            # Update C
            # reconstruct with updated Us
            coords = tl.tenalg.multi_mode_dot(Z.view(-1, self.c, self.s**2).float(), [Uc.T, Us.T], transpose=False, modes=[1, 2])
            Z_rec = tl.tenalg.multi_mode_dot(coords, [Uc, Us], transpose=False, modes=[1, 2])

            rec_loss = torch.mean(torch.norm(Z.view(-1, self.c, self.s**2).float() - Z_rec, p='fro', dim=[1, 2]) ** 2)
            rec_loss.backward()
            optimizerC.step()
            optimizerS.zero_grad()
            optimizerC.zero_grad()

            self.Us = Us
            self.Uc = Uc

            with torch.no_grad():
                if t % log_modulo == 0 and verbose:
                    print(f'Iteration {t} -- rec {rec_loss}')

                    noise = torch.Tensor(np.random.randn(batch_size, self.generator.z_space_dim)).to(self.device)
                    Z, x = self.sample(noise, layer=self.layer, partial=False)

                    plot_masks(Us.T, r=min(ranks[-1], 32), s=self.s)
                    plt.show()
                    plot_colours(x, Us.T, r=ranks[-1], s=self.s, seed=-1)
                    plt.show()

    def refine(self, Z, image, lr=1e-8, its=1000, log_modulo=250, verbose=True):
        """
        Performs the "refinement" step described in the paper, for a given sample Z.

        Parameters
        ----------
        Z : torch.Tensor
            Intermediate activations for target refinement.
        image : np.array
            Corresponding image for Z (purely for visualisation purposes).
        lr : float
            Learning rate the projected gradient descent.
        its : int
            Total number of iterations.
        log_modulo : int
            Parameter used to control how often "training" information is displayed.
        verbose : bool
            Prints extra information.

        Returns
        -------
        UsR : torch.Tensor
            The refined factors \tilde{P}_i.
        """
        np.random.seed(0)
        torch.manual_seed(0)

        #######################
        # init from global spatial factors
        UsR = self.Us.clone()
        Uc = self.Uc
        #######################

        zeros = torch.zeros_like(self.Us, device=self.device)
        for t in range(its):
            with torch.no_grad():
                z = Z.view(-1, self.c, self.s**2).float()
                # descend refinement term's gradient
                UsR_g = -4 * (torch.transpose(z,1,2)@Uc@Uc.T@z@UsR) + \
                    2 * (UsR@UsR.T@torch.transpose(z,1,2)@Uc@Uc.T@Uc@Uc.T@z@UsR + torch.transpose(z,1,2)@Uc@Uc.T@Uc@Uc.T@z@UsR@UsR.T@UsR)
                UsR_g = torch.sum(UsR_g, 0)

                # Update S
                UsR = UsR - lr * UsR_g
                # PGD step
                UsR = torch.maximum(UsR, zeros)

                if ((t + 1) % log_modulo == 0 and verbose):
                    print(f'iteration {t}')

                    plot_masks(UsR.T, s=self.s, r=min(self.ranks[-1], 16))
                    plt.show()
                    plot_colours(image, UsR.T, s=self.s, r=self.ranks[-1], seed=-1, alpha=0.9)
                    plt.show()

        return UsR

    def edit_at_layer(self, part, appearance, lam, t, Uc, Us, noise=None, b_idx=0):
        """
        Performs the "refinement" step described in the paper, for a given sample Z.

        Parameters
        ----------
        part : list
            List of ints containing the part(s) (column of Us) at which to edit.
        appearance : list
            List of ints containing the appearance (column of Uc) to apply at the corresponding part(s).
        lam : list
            List of ints containing the magnitude for each edit.
        t : int
            Random seed to edit
        Uc : np.array
            Learnt appearance factors
        Us : np.array
            Learnt parts factors
        noise : np.array
            If specified, the target latent code itself to edit (i.e. instead of providing than a random seed number).
        b_idx : int
            Index of biggan categories to use.

        Returns
        -------
        Z : torch.Tensor
            The intermediate activation at layer self.L
        image : np.array
            The original image for sample `t` or from latent code `noise`.
        image2 : np.array
            The edited image.
        part : np.array
            The part used to edit.
        """
        with torch.no_grad():
            if noise is None:
                np.random.seed(t)
                torch.manual_seed(t)
                noise = torch.Tensor(np.random.randn(1, self.generator.z_space_dim)).to(self.device)
            else:
                np.random.seed(0)
                torch.manual_seed(0)

            direc = 0
            for i in range(len(appearance)):
                a = Uc[:, appearance[i]]
                p = torch.sum(Us[:, part[i]], dim=-1).reshape([self.s, self.s])
                p = mapRange(p, torch.min(p), torch.max(p), 0.0, 1.0)

                # here, we basically form a rank-1 "tensor", to add to the target sample's activations.
                # intuitively, the non-zero spatial positions of the part are filled with the appearance vector.
                direc += lam[i] * tl.tenalg.outer([a, p])

            if self.gan_type in ['stylegan', 'stylegan2']:
                noise = self.generator.mapping(noise)['w']
                noise_trunc = self.generator.truncation(noise, trunc_psi=self.trunc_psi, trunc_layers=self.trunc_layers)

                Z = self.generator.synthesis(noise_trunc, start=self.start, stop=self.layer)['x']

                x = self.generator.synthesis(noise_trunc, x=Z, start=self.layer)['image']
                x_prime = self.generator.synthesis(noise_trunc, x=Z + direc, start=self.layer)['image']
            elif 'pggan' in self.gan_type:
                Z = self.generator(noise, start=self.start, stop=self.layer)['x']

                x = self.generator(Z, start=self.layer)['image']
                x_prime = self.generator(Z + direc, start=self.layer)['image']
            elif 'biggan' in self.gan_type:
                print(f'Choosing a {self.biggan_classes[b_idx]}')
                class_vector = torch.tensor(one_hot_from_names([self.biggan_classes[b_idx]]), device=self.device)
                noise_vector = torch.tensor(truncated_noise_sample(truncation=self.trunc_psi, batch_size=1, seed=t), device=self.device)

                result = self.generator(noise_vector, class_vector, self.trunc_psi, stop=self.layer)
                Z, cond_vector = result['z'], result['cond_vector']
                x = self.generator(Z, class_vector, self.trunc_psi, cond_vector=cond_vector, start=self.layer)['z']
                x_prime = self.generator(Z + direc, class_vector, self.trunc_psi, cond_vector=cond_vector, start=self.layer)['z']
            elif 'stylegan3' in self.gan_type:
                label = torch.zeros([1, 0], device=self.device)
                Z = self.generator(noise, label, stop=self.layer, truncation_psi=self.trunc_psi, noise_mode='const')

                x = self.generator(noise, label, x=Z, start=self.layer, stop=None, truncation_psi=self.trunc_psi, noise_mode='const')
                x_prime = self.generator(noise, label, x=Z + direc, start=self.layer, stop=None, truncation_psi=self.trunc_psi, noise_mode='const')

            image = np.array(Image.fromarray(postprocess(x.cpu().numpy())[0]).resize((256, 256)))
            image2 = np.array(Image.fromarray(postprocess(x_prime.cpu().numpy())[0]).resize((256, 256)))

            part = np.array(Image.fromarray(p.detach().cpu().numpy() * 255).convert('RGB').resize((256, 256), Image.NEAREST))
            return Z, image, image2, part

    def sample(self, noise, layer=5, partial=False, trunc_psi=1.0, trunc_layers=18, verbose=False):
        """
        Samples intermediate feature maps and resulting image the desired generator.

        Parameters
        ----------
        noise : np.array
            (batch_size, z_dim)-dim random standard gaussian noise.
        layer : int
            Intermediate layer at which to return intermediate features.
        partial : bool
            Perform full forward pass, and return image too? or just intermediate activations at layer number `layer`?
        trunc_psi : float
            Truncation value in [0, 1].
        trunc_layers : int
            Number of layers at which to apply truncation.
        biggan_classes : list
            List of strings specifying imagenet classes of interest (e.g. ['alp', 'breakwater']).
        verbose : bool
            Print out additional information?

        Returns
        -------
        Z : torch.Tensor
            The intermediate activations of shape [C, H, W].
        image : np.array
            Output RGB image.
        """
        with torch.no_grad():
            if self.gan_type in ['stylegan', 'stylegan2']:
                noise = self.generator.mapping(noise)['w']
                noise_trunc = self.generator.truncation(noise, trunc_psi=trunc_psi, trunc_layers=trunc_layers)
                Z = self.generator.synthesis(noise_trunc, start=self.start, stop=layer)['x']
                if not partial:
                    x = self.generator.synthesis(noise_trunc, x=Z, start=layer)['image']
            elif 'pggan' in self.gan_type:
                Z = self.generator(noise, start=self.start, stop=layer)['x']
                if not partial:
                    x = self.generator(Z, start=layer)['image']
            elif 'biggan' in self.gan_type:
                if verbose:
                    print(f'Using BigGAN class names: {", ".join(self.biggan_classes)}')

                class_vector = torch.tensor(one_hot_from_names(list(np.random.choice(self.biggan_classes, noise.shape[0])), batch_size=noise.shape[0]), device=self.device)
                noise_vector = torch.tensor(truncated_noise_sample(truncation=self.trunc_psi, batch_size=noise.shape[0]), device=self.device)

                result = self.generator(noise_vector, class_vector, self.trunc_psi, stop=layer)
                Z = result['z']
                cond_vector = result['cond_vector']

                if not partial:
                    x = self.generator(Z, class_vector, self.trunc_psi, cond_vector=cond_vector, start=layer)['z']
            elif 'stylegan3' in self.gan_type:
                label = torch.zeros([noise.shape[0], 0], device=self.device)
                if hasattr(self.generator.synthesis, 'input'):
                    m = np.linalg.inv(make_transform((0,0), 0))
                    self.generator.synthesis.input.transform.copy_(torch.from_numpy(m))

                Z = self.generator(noise, label, x=None, start=0, stop=layer, truncation_psi=trunc_psi, noise_mode='const')
                if not partial:
                    x = self.generator(noise, label, x=Z, start=layer, stop=None, truncation_psi=trunc_psi, noise_mode='const')

            if verbose:
                print(f'-- Partial Z shape at layer {layer}: {Z.shape}')

            if partial:
                return Z, None
            else:
                image = postprocess(x.detach().cpu().numpy())
                image = np.array(Image.fromarray(image[0]).resize((256, 256)))
                return Z, image

    def save(self):
        Uc_path = f'./checkpoints/Uc-name_{self.model_name}-layer_{self.layer}-rank_{self.ranks[0]}.npy'
        Us_path = f'./checkpoints/Us-name_{self.model_name}-layer_{self.layer}-rank_{self.ranks[1]}.npy'

        np.save(Us_path, self.Us.detach().cpu().numpy())
        np.save(Uc_path, self.Uc.detach().cpu().numpy())

        print(f'Saved factors to {Us_path}, {Uc_path}')