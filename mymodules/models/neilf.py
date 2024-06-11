# neilf.py
""" This script includes some models from NeILF.
Some lines may refer to: https://github.com/apple/ml-neilf.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import numpy as np

import torch
import torch.nn as nn

from mymodules.embedders import Embedder


class SIRENLayer(nn.Module):
    """ SIREN layer [Sitzmann et al. 2020].

    For the implementation, refer to:
        https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/model/nn_arch.py#L10.
    The original paper is: https://arxiv.org/pdf/2006.09661.pdf.

    Attributes:
        in_num (int): the number of input dimension.
        omega_o (float): a hyperparameter for initialization. For more details, see Sec. 3.2 of the original paper.
    """

    def __init__(self,
                 in_num: int,
                 out_num: int,
                 use_bias: bool = True,
                 is_first_layer: bool = False,
                 omega_o: float = 30.):
        super(SIRENLayer, self).__init__()

        self.in_num = in_num
        self.omega_o = omega_o

        self.linear = nn.Linear(in_num, out_num, bias=use_bias)

        # initialize weights of the linear layer.
        # See Sec 3.2 of the original paper.
        if is_first_layer:
            nn.init.uniform_(self.linear.weight, -1 / self.in_num * self.omega_o, 1 / self.in_num * self.omega_o)
        else:
            nn.init.uniform_(self.linear.weight, -np.sqrt(3 / self.in_num), np.sqrt(3 / self.in_num))
        nn.init.zeros_(self.linear.bias)

    def forward(self, inx: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(inx))


class BRDFNet(nn.Module):
    """ MLP for estimating BRDF.

    Refer to: https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/model/nn_arch.py#L46

    Attributes:
        depth (int): the depth of MLP.
        width (int): the width of MLP.
        in_ch (int): the number of the channel of input.
        out_ch (int): the number of the channel of output.
        skips (list): add skip connections in the designated layers. note that the first layer is counted as zero.
        embedder (myclasses.embedder.Embedder): embedder.
        last_activation_func (nn.modules.activation): activation function for the output. if empty, no activation.
    """

    def __init__(self,
                 depth: int,
                 width: int,
                 in_ch: int,
                 out_ch: int,
                 skips: list,
                 embedder: Embedder,
                 weight_norm: bool = False):
        super(BRDFNet, self).__init__()

        self.depth = depth
        self.width = width
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skips = skips
        self.embedder = embedder
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
        inx = self.embedder.embed(inx)  # positional encoding

        x = inx
        for i in range(self.depth):
            layer = getattr(self, "layer_{}".format(i))

            if i in self.skips:
                x = torch.cat([x, inx], dim=1)

            x = layer(x)

        x = self.last_activation_func(x)  # Note: SIREN layer already includes activation.

        return x


class NeILFNet(nn.Module):
    """ this is an MLP for estimating a position-dependent lighting.
    Refer to: https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/model/nn_arch.py#L114

    Attributes:
        depth (int): the depth of MLP.
        width (int): the width of MLP.
        in_ch (int): the number of the channel of input.
        out_ch (int): the number of the channel of output.
        skips (list): add skip connections in the designated layers. note that the first layer is counted as zero.
        embedder (myclasses.embedder.Embedder): embedder.
        last_activation_func : activation function for the output. if empty, no activation.
    """
    def __init__(self,
                 depth: int,
                 width: int,
                 in_ch: int,
                 out_ch: int,
                 skips: list,
                 embedder: Embedder,
                 weight_norm: bool = False):
        super(NeILFNet, self).__init__()

        self.depth = depth
        self.width = width
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skips = skips
        self.embedder = embedder
        self.last_activation_func = torch.exp

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
        pos = inx[:, 0:3]
        view_embed = self.embedder.embed(inx[:, 3:6])

        x = view_embed
        for i in range(self.depth):
            layer = getattr(self, "layer_{}".format(i))

            if i in self.skips:
                x = torch.cat([x, view_embed, pos], dim=1)

            x = layer(x)

        x = self.last_activation_func(x)  # Note: SIREN layer already includes activation.

        return x


class NeILFNetWoPos(nn.Module):
    """ this is an MLP for estimating a position-independent lighting.
    Refer to: https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/model/nn_arch.py#L114

    Attributes:
        depth (int): the depth of MLP.
        width (int): the width of MLP.
        in_ch (int): the number of the channel of input.
        out_ch (int): the number of the channel of output.
        skips (list): add skip connections in the designated layers. note that the first layer is counted as zero.
        embedder (myclasses.embedder.Embedder): embedder.
        last_activation_func : activation function for the output. if empty, no activation.
    """
    def __init__(self,
                 depth: int,
                 width: int,
                 in_ch: int,
                 out_ch: int,
                 skips: list,
                 embedder: Embedder,
                 weight_norm: bool = False):
        super(NeILFNetWoPos, self).__init__()

        self.depth = depth
        self.width = width
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skips = skips
        self.embedder = embedder
        self.last_activation_func = torch.exp

        for i in range(depth):
            if i == 0:  # initial layer.
                layer = SIRENLayer(in_ch, width, is_first_layer=True)

            elif i == depth - 1:  # last layer.
                layer = nn.Linear(width, out_ch)
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, np.log(1.5))  # ?

            elif i in skips:  # this layer contains a skip connection.
                layer = SIRENLayer(in_ch + width, width)  # pos (3,) + view_embedded (emb_out,) + width

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
        inx = self.embedder.embed(inx)  # positional encoding
        x = inx
        for i in range(self.depth):
            layer = getattr(self, "layer_{}".format(i))

            if i in self.skips:
                x = torch.cat([x, inx], dim=1)

            x = layer(x)

        x = self.last_activation_func(x)  # Note: SIREN layer already includes activation.

        return x
