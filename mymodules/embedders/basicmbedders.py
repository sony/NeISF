# embedders.py
""" This script defines some embedders.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

Notes:
    some lines may refer to the followings:
    - https://github.com/bmild/nerf.
    - https://github.com/lioryariv/volsdf/blob/main/code/model/embedder.py.
    - https://github.com/jmclong/random-fourier-features-pytorch.
"""

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


class Embedder:
    def __init__(self):
        self.out_num = None

    def embed(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError


class PEEmbedder(Embedder):
    r""" Positional Encoding (PE) from volSDF.

    Attributes:
        l_num (int): l_num of PE (https://arxiv.org/pdf/2003.08934.pdf, Eq. 4).
            :math:`PE(p) = [p, \sin(2^0 \pi p), \cos(2^0 \pi p), ..., \sin(2^(l_num-1) \pi p), \cos(2^(l_num-1) \pi p)]`
        input_dim (int): the number of dimensions of the input.
        include_input (bool): if you want to include the input or not.
        embed_fns (list): a list of functions.
        out_dim (int): the dimension after PE.
    """

    def __init__(self, l_num: int, input_dim: int = 3, include_input: bool = True):
        super(PEEmbedder, self).__init__()

        self.l_num = l_num
        self.input_dim = input_dim
        self.include_input = include_input

        embed_fns, out_dim = self.generate_embedding_fn()
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def generate_embedding_fn(self) -> Tuple[list, int]:
        embed_fns = []
        out_dim = 0

        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dim

        freq_bands = 2. ** torch.linspace(0., self.l_num - 1, self.l_num)

        for freq_ in freq_bands:
            embed_fns.append(lambda x, freq=freq_: torch.sin(x * freq))
            embed_fns.append(lambda x, freq=freq_: torch.cos(x * freq))

            out_dim += 2 * self.input_dim

        return embed_fns, out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FourierFeatureEmbedder(Embedder):
    """ Gaussian encoding of random Fourier features.

    For the idea behind, refer to: https://arxiv.org/pdf/2006.10739.pdf, Eq. 5.
    For the implementation, refer to: https://github.com/jmclong/random-fourier-features-pytorch/blob/main/rff/layers.py.
    Based on the paper's knowledge, we also use a = 1, and sample b from Gaussian distribution.

    Attributes:
        b (torch.Tensor): a matrix sampled from Gaussian distribution (out_size, in_size).
        out_dim (int): the output dimension.
    """

    def __init__(self, in_size: int, enc_size: int, sigma: float, device: torch.device):
        """
        Args:
            in_size (int): the number of input dimension.
            enc_size (int): the number of dimensions the B matrix maps to.
            sigma (float): sigma of Gaussian distribution.
            device (torch.device): the device you want to use.
        """

        super(FourierFeatureEmbedder, self).__init__()

        self.b = self.sample_b(sigma, (enc_size, in_size), device)
        self.out_dim = 2 * enc_size

    def embed(self, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v (torch.Tensor): input tensor (n, 3).
        """
        return gaussian_embed_jit(v, self.b)

    @staticmethod
    def sample_b(sigma: float, size: tuple, device: torch.device) -> torch.Tensor:
        r"""Matrix of size :attr:`size` sampled from from :math:`\mathcal{N}(0, \sigma^2)`
        Args:
            sigma (float): standard deviation.
            size (tuple): size of the matrix sampled.
            device (torch.device): the device you want to use.
        """
        return torch.randn(size, device=device) * sigma


@torch.jit.script
def gaussian_embed_jit(v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

        Args:
            v (Tensor): input tensor of shape (n, 3).
            b (Tensor): projection matrix of shape (o, 3)
        Returns:
            Tensor: mapped tensor of shape (n, o).
    """
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
