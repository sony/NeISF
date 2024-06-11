# tensorutils.py
""" Library for Tensor operations.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import torch
from torch import Tensor

EPS = 1e-07


def normalize_one_dim_tensor(v: Tensor) -> Tensor:
    """ Normalize 1D tensor.

    Args:
        v (Tensor): 1D tensor (3,).

    Returns:
        Normalized 1D tensor (3,).
    """

    if v.ndim != 1 or v.shape[0] != 3:
        raise ValueError("input tensor's shape must be (3,)")

    return v / (torch.norm(v) + EPS)


def normalize_tensor(v: Tensor) -> Tensor:
    """ Normalize tensors of shape (n, 3).

    Args:
        v (Tensor): a tensor of shape (n, 3).

    Return:
        Normalized tensor of shape (n, 3).
    """

    if v.ndim != 2:
        raise ValueError("vector dim must be 2")
    if v.shape[1] != 3:
        raise ValueError("vector second dim size must be 3")

    return v / (torch.norm(v, dim=1, keepdim=True) + EPS)


def dot_product_tensor(vec1: Tensor, vec2: Tensor) -> Tensor:
    """ Compute dot product of two tensors.

    Args:
        vec1 (Tensor): the first tensor of shape (n, 3).
        vec2 (Tensor): the second tensor of shape (n, 3).

    Returns:
        Computed dot product of the two tensors (n, 1).
    """

    if vec1.ndim != 2 or vec2.ndim != 2:
        raise ValueError("vector dim must be 2")
    if vec1.shape[1] != 3 or vec2.shape[1] != 3:
        raise ValueError("vector second dim size must be 3")

    return torch.sum(vec1 * vec2, dim=1, keepdim=True)


def calc_rot_mat_from_two_vec(vec1: Tensor, vec2: Tensor) -> Tensor:
    """ Compute a rotation matrix of shape (n, 3, 3), which rotates the second input Tensor to the first one.

    For the implementation, refer to:
        https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311

    Args:
        vec1 (Tensor): a tensor of shape (n, 3).
        vec2 (Tensor): a tensor of shape (n, 3).

    Returns:
        Computed rotation matrix of shape (n, 3, 3).
    """

    b_size = vec1.shape[0]

    v = torch.cross(vec1, vec2)

    cos = torch.bmm(vec1.view(b_size, 1, 3), vec2.view(b_size, 3, 1))
    cos = cos.reshape(b_size, 1, 1).repeat(1, 3, 3)

    skew_sym_mat = torch.zeros(b_size, 3, 3).to(vec1.device)
    skew_sym_mat[:, 0, 1] = -v[:, 2]
    skew_sym_mat[:, 0, 2] = v[:, 1]
    skew_sym_mat[:, 1, 0] = v[:, 2]
    skew_sym_mat[:, 1, 2] = -v[:, 0]
    skew_sym_mat[:, 2, 0] = -v[:, 1]
    skew_sym_mat[:, 2, 1] = v[:, 0]

    identity_mat = torch.zeros(b_size, 3, 3).to(vec1.device)
    identity_mat[:, 0, 0] = 1
    identity_mat[:, 1, 1] = 1
    identity_mat[:, 2, 2] = 1

    rot_mat = identity_mat + skew_sym_mat
    rot_mat = rot_mat + torch.bmm(skew_sym_mat, skew_sym_mat) / (1 + cos).clamp(min=1e-7)
    zero_cos_loc = (cos == -1).float()
    rot_mat_inv = torch.zeros(b_size, 3, 3).to(vec1.device)
    rot_mat_inv[:, 0, 0] = -1
    rot_mat_inv[:, 1, 1] = -1
    rot_mat_inv[:, 2, 2] = -1

    out = rot_mat * (1 - zero_cos_loc) + rot_mat_inv * zero_cos_loc  # (n, 3, 3)

    return out


def sigmoid(xs: Tensor, alpha: float, center: float) -> Tensor:
    r""" Computes :math:`\frac{1}{1 + \exp(-\alpha \times (x - center))}`.

    Args:
        xs (Tensor): the values whose sigmoid is required.
        alpha (float): a value that defines the gradient of the sigmoid.
        center (float): a value that defines the shift along x-axis.

    Returns:
        Computed sigmoid. 1 / (1 + torch.exp(-alpha * (xs - center)))
    """

    if alpha <= 0:
        raise ValueError("Alpha must be larger than 0.")

    sigmoid_range = 34.538776394910684

    x = torch.clamp(-alpha * (xs - center), -sigmoid_range, sigmoid_range)
    return 1 / (1 + torch.exp(x))


def calc_weighted_sum(weights: Tensor, target: Tensor) -> Tensor:
    """ Computes a weighted sum of the target tensor.

    Args:
        weights (Tensor): Weight tensor of shape (a, b).
        target (Tensor): Target tensor of shape (a * b, c).

    Returns:
        Computed weighted sum of shape (a, c).
    """

    if weights.ndim != 2 or target.ndim != 2:
        raise ValueError("the ndim of the input tensors must be 2.")
    if target.shape[0] != weights.shape[0] * weights.shape[1]:
        raise ValueError("the shape of weights and target do not match.")

    target_ = target.reshape(-1, weights.shape[1], target.shape[-1])  # (a, b, c)
    weighted = weights.unsqueeze(-1) * target_  # (a, b, 1) * (a, b, c) = (a, b, c)
    weighted_sum = torch.sum(weighted, dim=1, keepdim=False)  # (a, c)

    return weighted_sum


def gamma(x: Tensor, gamma_val: float = 2.2) -> Tensor:
    """ Compute the gamma correction of the input tensor.

    Args:
        x (Tensor): input tensor in any shape.
        gamma_val (float): gamma value.

    Returns:
        (Tensor): gamma corrected tensor.
    """

    x = torch.clip(x, min=0.) ** (1 / gamma_val)
    return torch.clip(x, 0., 1.)
