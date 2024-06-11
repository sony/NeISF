# volsdf.py
""" This script defines some networks introduced by volSDF [Yariv et al. 2021].
Some lines may refer to: https://github.com/lioryariv/volsdf/blob/main/code/model/network.py.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from mymodules.embedders import Embedder


class LaplaceDensity(nn.Module):
    """ Class for Laplace density.
    Refer to: https://github.com/lioryariv/volsdf/blob/main/code/model/density.py.
    """

    def __init__(self, beta, beta_min):
        super(LaplaceDensity, self).__init__()

        self.beta = nn.Parameter(torch.tensor(beta))  # nn.Parameter contains the gradient. Updated during training.
        self.beta_min = torch.tensor(beta_min)

    def forward(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        return self.beta.abs() + self.beta_min


class SDFNet(nn.Module):
    """ An MLP which computes an SDF.
    Refer to: https://github.com/lioryariv/volsdf/blob/main/code/model/network.py.

    Attributes:
        depth (int): the depth of MLP.
        width (int): the width of MLP.
        in_ch (int): the number of the channel of input.
        out_ch (int): the number of the channel of output.
        skips (list): add skip connections in the designated layers. note that the first layer is counted as zero.
        embedder (Embedder): embedder.
        sphere_scale (float): used when clipping the sdf values. if not to clip the sdf, use -1. otherwise,
            must be larger than zero. this hyperparameter is not written in the original paper. for more details,
            refer to https://github.com/lioryariv/volsdf/blob/main/code/model/network.py.
        bias (float): used for initializing the last layer's bias.
                refer to https://github.com/lioryariv/volsdf/blob/main/code/confs/dtu.conf.
        activation (nn.modules.activation.Softplus): softplus activation.
        density (LaplaceDensity): Laplace density.
    """

    def __init__(self,
                 depth: int,
                 width: int,
                 in_ch: int,
                 out_ch: int,
                 skips: list,
                 embedder: Embedder,
                 sphere_scale: float = 1.0,
                 bias: float = 0.6,
                 geometric_init: bool = True,
                 weight_norm: bool = True):
        """
        Args:
            geometric_init (bool): ``true`` for geometric init (https://arxiv.org/abs/1911.10414).
            weight_norm (bool): ``true`` for weight_norm.
        """
        super(SDFNet, self).__init__()

        if min(skips) <= 0 or max(skips) >= depth - 1:
            raise ValueError("Your designated skip layers may include the wrong numbers.")

        if sphere_scale <= 0 and sphere_scale != -1.:
            raise ValueError("sphere_scale must be -1 or larger than zero.")

        self.depth = depth
        self.width = width
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skips = skips
        self.embedder = embedder
        self.sphere_scale = sphere_scale
        self.bias = bias
        self.activation = nn.Softplus(beta=100)
        self.density = LaplaceDensity(beta=0.1, beta_min=0.0001)

        for i in range(depth):
            if i == 0:  # initial layer.
                layer = nn.Linear(in_ch, width)
                if geometric_init:
                    # geometric initialization.
                    # Refer to: https://arxiv.org/abs/1911.10414
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.constant_(layer.weight[:, 3:], 0.0)
                    nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(width))

            elif i == depth - 1:  # last layer.
                layer = nn.Linear(width, out_ch)
                if geometric_init:
                    # geometric initialization.
                    nn.init.constant_(layer.bias, -self.bias)
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(width), std=0.0001)

            elif i in skips:
                layer = nn.Linear(in_ch + width, width)
                if geometric_init:
                    # geometric initialization.
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(width))
                    nn.init.constant_(layer.weight[:, -(in_ch - 3):], 0.0)

            else:
                layer = nn.Linear(width, width)
                if geometric_init:
                    # geometric initialization.
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(width))

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            setattr(self, "linear_{}".format(i), layer)

    def forward(self, inx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inx (torch.Tensor): input.
        Returns:
            (torch.Tensor): output.
        """
        inx = self.embedder.embed(inx)  # positional encoding

        x = inx

        for i in range(self.depth):
            layer = getattr(self, "linear_{}".format(i))

            if i in self.skips:
                x = torch.cat([x, inx], dim=1) / np.sqrt(2)

            x = layer(x)

            if i < self.depth - 1:  # not the last layer, activate the output.
                x = self.activation(x)

        return x

    def get_all_outputs(self,
                        x: torch.Tensor,
                        bounding_sphere_r: float,
                        is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ compute sdf value, feature vectors, and gradients.

        Args:
            x (torch.Tensor): input tensor.
            bounding_sphere_r (float): radius of the bounding sphere.
            is_training (bool): flag for handling gradients.
        Returns:
            sdf value, feature vectors, and gradients.
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            output = self.forward(x)

            sdf_val = output[:, :1]
            feature_vec = output[:, 1:]

            sdf_val = self.clamp_sdf_by_sphere(x, sdf_val, bounding_sphere_r)

            # Compute gradient
            d_output = torch.ones_like(sdf_val, requires_grad=False, device=sdf_val.device)
            gradients = torch.autograd.grad(
                outputs=sdf_val,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

        if not is_training:  # testing.
            sdf_val = sdf_val.detach()
            feature_vec = feature_vec.detach()
            gradients = gradients.detach()

        return sdf_val, feature_vec, gradients

    def get_sdf_vals(self, x: torch.Tensor, bounding_sphere_r: float) -> torch.Tensor:
        """ return sdf values.

        Args:
            x (torch.Tensor): input tensor.
            bounding_sphere_r (float): radius of the bounding sphere.
        Returns:
            (torch.Tensor): sdf values.
        """
        sdf_val = self.forward(x)[:, :1]
        return self.clamp_sdf_by_sphere(x, sdf_val, bounding_sphere_r)

    def get_gradient(self, x: torch.Tensor, bounding_sphere_r: float) -> torch.Tensor:
        """ return gradients.

        Args:
            x (torch.Tensor): input tensor.
            bounding_sphere_r (float): radius of the bounding sphere.
        Returns:
            (torch.Tensor): gradients.
        """
        with torch.enable_grad():
            x.requires_grad_(True)

            sdf_val = self.forward(x)[:, :1]
            sdf_val = self.clamp_sdf_by_sphere(x, sdf_val, bounding_sphere_r)

            # Compute gradient
            d_output = torch.ones_like(sdf_val, requires_grad=False, device=sdf_val.device)
            gradients = torch.autograd.grad(
                outputs=sdf_val,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

        return gradients

    def clamp_sdf_by_sphere(self, x: torch.Tensor, sdf_val: torch.Tensor, bounding_sphere_r: float) -> torch.Tensor:
        """ Clamping the SDF with the scene bounding sphere so that all the rays are eventually occluded.

        Args:
            x (torch.Tensor): input tensor.
            sdf_val (torch.Tensor): sdf values.
            bounding_sphere_r (float): radius of the bounding sphere.
        Returns:
            (torch.Tensor): clamped sdf.
        """
        return_val = sdf_val

        if self.sphere_scale != -1.:
            sphere_sdf = self.sphere_scale * (bounding_sphere_r - x.norm(2, 1, keepdim=True))
            return_val = torch.minimum(sdf_val, sphere_sdf)

        return return_val


class RenderNet(nn.Module):
    """ A rendering network used in volsdf.
    Refer to: https://github.com/lioryariv/volsdf/blob/main/code/model/network.py.

    Attributes:
        depth (int): the number of layers.
        width (int): the number of nodes a layer contains.
        in_ch (int): the number of input channels.
        out_ch (int): the number of output channels.
        embedder (myclasses.embedder.Embedder): embedder.
        relu (nn.modules.activation.ReLU): ReLU function.
        last_activation (nn.modules.activation): the last activation function.
        mode (str): "fg" for foreground, "bg" for background. raise ValueError for others.
    """
    def __init__(self,
                 depth: int,
                 width: int,
                 in_ch: int,
                 out_ch: int,
                 embedder: Embedder,
                 weight_norm: bool = True,
                 last_activation: nn.modules.activation = nn.Sigmoid(),
                 mode: str = "fg"):
        super(RenderNet, self).__init__()

        self.depth = depth
        self.width = width
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.embedder = embedder

        self.relu = nn.ReLU()
        self.last_activation = last_activation

        if mode == "fg" or mode == "bg":
            self.mode = mode
        else:
            raise ValueError("you're using a wrong mode.")

        for i in range(depth):
            if i == 0:  # initial layer.
                layer = nn.Linear(in_ch, width)
            elif i == depth - 1:  # last layer.
                layer = nn.Linear(width, out_ch)
            else:
                layer = nn.Linear(width, width)

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            setattr(self, "linear_{}".format(i), layer)

    def forward(self,
                points: torch.Tensor,
                normals: torch.Tensor,
                view_dirs: torch.Tensor,
                feature_vectors: torch.Tensor) -> torch.Tensor:
        view_dirs = self.embedder.embed(view_dirs)

        x = None
        if self.mode == "fg":
            x = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == "bg":
            x = torch.cat([view_dirs, feature_vectors], dim=-1)

        for i in range(self.depth):
            layer = getattr(self, "linear_{}".format(i))
            x = layer(x)

            if i < self.depth - 1:
                x = self.relu(x)

        return self.last_activation(x)
