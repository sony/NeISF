# neisf.py
""" This script includes some models from NeILF.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import numpy as np

import torch
import torch.nn as nn

from .neilf import SIRENLayer


class IncidentNet(nn.Module):
    """ this is an MLP for estimating a position-dependent stokes vector.
    Implementation is modified from: https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/model/nn_arch.py#L114

    Attributes:
        depth (int): the depth of MLP.
        width (int): the width of MLP.
        in_ch (int): the number of the channel of input.
        out_ch (int): the number of the channel of output.
        skips (list): add skip connections in the designated layers. note that the first layer is counted as zero.
        last_activation_func : activation function for the output. if empty, no activation.
    """
    def __init__(self,
                 depth: int,
                 width: int,
                 in_ch: int,
                 out_ch: int,
                 skips: list,
                 last_activation_func,
                 weight_norm: bool = False):
        super(IncidentNet, self).__init__()

        self.depth = depth
        self.width = width
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skips = skips
        self.last_activation_func = last_activation_func

        for i in range(depth):
            if i == 0:  # initial layer.
                layer = SIRENLayer(in_ch, width, is_first_layer=True)

            elif i == depth - 1:  # last layer.
                layer = nn.Linear(width, out_ch)
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, np.log(1.5))  # ?

            elif i in skips:  # this layer contains a skip connection.
                layer = SIRENLayer(3 + in_ch + width, width)  # pos (3,) + view_embedded (emb_out,) + width

            else:
                layer = SIRENLayer(width, width)

            if weight_norm:
                if isinstance(layer, SIRENLayer):
                    layer.linear = nn.utils.weight_norm(layer.linear)
                else:
                    raise ValueError("all the weights are initialized to be zero. "
                                     "weight normalization could produce`nan`value.")

            setattr(self, "layer_{}".format(i), layer)

    def forward(self, inx: torch.Tensor) -> torch.Tensor:
        x = inx
        for i in range(self.depth):
            layer = getattr(self, "layer_{}".format(i))

            if i in self.skips:
                x = torch.cat([x, inx], dim=1)

            x = layer(x)

        x = self.last_activation_func(x)  # Note: SIREN layer already includes activation.

        return x


class BRDFNet(nn.Module):
    """ this is an MLP for estimating BRDF.
    Refer to: https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/model/nn_arch.py#L46

    Attributes:
        depth (int): the depth of MLP.
        width (int): the width of MLP.
        in_ch (int): the number of the channel of input.
        out_ch (int): the number of the channel of output.
        skips (list): add skip connections in the designated layers. note that the first layer is counted as zero.
        last_activation_func (nn.modules.activation): activation function for the output. if empty, no activation.
    """
    def __init__(self,
                 depth: int,
                 width: int,
                 in_ch: int,
                 out_ch: int,
                 skips: list,
                 weight_norm: bool = False):
        super(BRDFNet, self).__init__()

        self.depth = depth
        self.width = width
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skips = skips
        self.last_activation_func = nn.Tanh()

        for i in range(depth):
            if i == 0:  # initial layer.
                layer = SIRENLayer(in_ch, width, is_first_layer=True)

            elif i == depth - 1:  # last layer.
                layer = nn.Linear(width, out_ch)
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

            elif i in skips:  # this layer contains a skip connection.
                layer = SIRENLayer(in_ch + width, width)

            else:
                layer = SIRENLayer(width, width)

            if weight_norm:
                if isinstance(layer, SIRENLayer):
                    layer.linear = nn.utils.weight_norm(layer.linear)
                else:
                    raise ValueError("all the weights are initialized to be zero. "
                                     "weight normalization could produce`nan`value.")

            setattr(self, "layer_{}".format(i), layer)

    def forward(self, inx: torch.Tensor) -> torch.Tensor:
        x = inx
        for i in range(self.depth):
            layer = getattr(self, "layer_{}".format(i))

            if i in self.skips:
                x = torch.cat([x, inx], dim=1)

            x = layer(x)

        x = self.last_activation_func(x)  # Note: SIREN layer already includes activation.

        return x
