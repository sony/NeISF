# sampler_base.py
""" Base class for the samplers.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import abc
from typing import Any

import torch


class SamplerBase(metaclass=abc.ABCMeta):
    """ Base class of position sampler.

    Attributes:
        near (float): near bound of the sampling positions.
        far (float):  far bound of the sampling positions.
        n_samples (int): the number of samples for each ray.
        device (torch.device): the device you will use for training.
        bounding_sphere_r (float): radius of the bounding sphere.
        use_background (bool):
            When you want to separately model foreground and background of the scene, use ``True``.
            Refer to: https://arxiv.org/pdf/2010.07492.pdf sec. 4.
    """

    def __init__(self,
                 near: float,
                 far: float,
                 n_samples: int,
                 device: torch.device,
                 bounding_sphere_r: float,
                 use_background: bool):

        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.device = device
        self.bounding_sphere_r = bounding_sphere_r
        self.use_background = use_background

    def get_z_vals(self, *args, **kwargs) -> Any:
        raise NotImplementedError
