# basemodels.py
""" This script includes some basic models.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import torch.nn as nn


class MLP(nn.Module):
    """ Basic MLP.

    Attributes:
        depth (int): the number of layers.
        width (int): the number of nodes a layer contains.
        in_ch (int): the number of input channels.
        out_ch (int): the number of output channels.
        relu (nn.modules.activation.ReLU): ReLU function.
        last_activation_func (nn.modules.activation): activation function for the output. if empty, no activation.
        use_weight_norm (bool): if True, nn.utils.weight_norm() will be used for the weight normalization.
    """

    def __init__(self, depth: int, width: int, in_ch: int, out_ch: int,
                 last_activation_func=None, use_weight_norm: bool = False):

        super(MLP, self).__init__()

        self.depth = depth
        self.width = width
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.relu = nn.ReLU()
        self.last_activation_func = last_activation_func
        self.use_weight_norm = use_weight_norm

        # Set up the MLP.
        for i in range(depth):
            if i == 0:  # Initial layer.
                layer = nn.Linear(in_ch, width)
            elif i == depth - 1:  # Last layer.
                layer = nn.Linear(width, out_ch)
            else:
                layer = nn.Linear(width, width)

            if use_weight_norm:
                nn.utils.weight_norm(layer)

            setattr(self, "linear_{}".format(i), layer)

    def forward(self, x):
        for i in range(self.depth):
            layer = getattr(self, "linear_{}".format(i))
            x = layer(x)

            if i < self.depth - 1:
                x = self.relu(x)

        if self.last_activation_func is not None:
            x = self.last_activation_func(x)

        return x
