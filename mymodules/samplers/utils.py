# samplers/utils.py
""" Library for some utility functions for the samplers.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import torch
from torch import Tensor

from mymodules.tensorutils import dot_product_tensor


def get_sphere_intersections(rays_o: Tensor, rays_d: Tensor, r: float) -> Tensor:
    """ Computes the intersection with the rays and the bounding sphere.

    Args:
        rays_o (Tensor): the origin of rays (b_size, 3).
        rays_d (Tensor): the direction of rays (b_size, 3).
        r (float): radius of the bounding sphere.

    Returns:
        (Tensor): computed intersections (b_size, 1).

    Note:
        This function assumes that the cameras are inside the sphere.
        This function is solving the following equation for t.
        1. a = rays_o + t * rays_d (t > 0).
        2. ||a|| = r.
        Because we are assuming t must be larger or equal to zero, we discard the smaller solution.
        Her we do not check if the cameras are located inside the sphere.
    """

    ray_cam_dot = dot_product_tensor(rays_d, rays_o)

    under_sqrt = ray_cam_dot ** 2 - (torch.sum(rays_o * rays_o, dim=1, keepdim=True) - r ** 2)  # always > 0.

    t = torch.sqrt(under_sqrt) - ray_cam_dot  # (b_size, 1)
    t = torch.clamp(t, 0.0)

    return t
