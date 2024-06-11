# losses.py
""" Library for loss functions.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import torch
import torch.nn as nn
from torch import Tensor


class EikonalLoss(nn.Module):
    """ Eikonal loss to force a network to output SDF.
    Refer to: https://github.com/lioryariv/volsdf/blob/main/code/model/loss.py
    """

    def __init__(self):
        super(EikonalLoss, self).__init__()

    @staticmethod
    def forward(grad_theta: Tensor) -> Tensor:
        return ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()


class WeightedL1Loss(nn.Module):
    """ This loss function calculates L1 loss considering the transmittance of the background.
    """

    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    @staticmethod
    def forward(x1: Tensor, x2: Tensor, bg_transmittance: Tensor) -> Tensor:
        """
        Args:
            x1 (Tensor): tensor of shape (n, 3).
            x2 (Tensor): tensor of shape (n, 3).
            bg_transmittance (Tensor): transmittance of BG. 1 for BG pixels, 0 for others (n, 1).
        """
        diff = torch.abs(x1 - x2)  # (n, 3)
        diff = diff * (1 - bg_transmittance)  # (n, 3)

        return diff.mean()


class MaskedL1Loss(nn.Module):
    """ This loss function calculates L1 loss according to the input mask.
    """
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.loss_func = nn.L1Loss()

    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x1 (Tensor): input tensor 1 (n, 3).
            x2 (Tensor): input tensor 2 (n, 3).
            mask (Tensor): mask tensor. must be bool (n, 3).
        """

        x1_ = x1[mask]
        x2_ = x2[mask]

        return self.loss_func(x1_, x2_)
