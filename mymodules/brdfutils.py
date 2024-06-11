# brdfutils.py
""" Library for Bidirectional Reflectance Distribution Function (BRDF).

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import math

import torch
from torch import Tensor

from .tensorutils import normalize_tensor, dot_product_tensor

EPS = 1e-07


def calc_chi(x: Tensor) -> Tensor:
    """ Compute chi factor.
    The definition of chi is: f(x) = 1 (x>0), f(X) = 0 (else).

    Args:
        x (Tensor): input.

    Returns:
        (Tensor): chi factor.
    """
    return torch.clip(torch.sign(x), 0., 1.)


def calc_normal_distribution_walter(n_dot_h: Tensor, roughness_sq: Tensor) -> Tensor:
    """ Compute distribution of surface-normal based on [Walter et al. 2007].

    Refer to: http://vclab.kaist.ac.kr/siggraphasia2018p2/polarization_supple_1.pdf.

    Args:
        n_dot_h (Tensor): dot product of normal and half vector (n, 1).
        roughness_sq (Tensor): {surface roughness}^2 (n, 1).

    Returns:
        (Tensor): surface normal distribution based on [Walter et al. 2007] (n, 1).
    """

    cos2_th = n_dot_h * n_dot_h
    tan2_th = (1 - cos2_th) / (cos2_th + EPS)
    cos4_th = cos2_th * cos2_th

    return roughness_sq * calc_chi(n_dot_h) / (math.pi * cos4_th * (roughness_sq + tan2_th) ** 2 + EPS)


def calc_normal_distribution_neilf(n_dot_h: Tensor, roughness_sq: Tensor) -> Tensor:
    """ Compute distribution of surface-normal defined on NeILF [Yao et al. 2022].

    Refer to: https://arxiv.org/abs/2203.07182v2, Supplementary material 1.2.

    The explanation above may contain a mistake. Please refer to the following:
        https://github.com/apple/ml-neilf/blob/main/code/model/neilf_brdf.py#L153.

    Args:
        n_dot_h (Tensor): dot product of normal and half vector (n, 1).
        roughness_sq (Tensor): {surface roughness}^2 (n, 1).

    Returns:
        (Tensor): surface normal distribution based on GGX model (n, 1).
    """

    roughness_sq_ = roughness_sq + EPS
    k = 2 * (n_dot_h - 1) / roughness_sq_

    return torch.exp(k) / math.pi / roughness_sq_


def calc_lambda_heitz(x_dot_n: Tensor, roughness_sq: Tensor):
    """ Compute the lamda based on [Heitz 2014].

    Refer to: http://vclab.kaist.ac.kr/siggraphasia2018p2/polarization_supple_1.pdf.

    Args:
        x_dot_n (Tensor): dot product of x and n (n, 1).
        roughness_sq (Tensor): {surface roughness}^2 (n, 1).

    Returns:
        Computed lamda of shape (n, 1).
    """

    cos2_th = x_dot_n * x_dot_n
    tan2_th = (1 - cos2_th) / (cos2_th + EPS)

    return 2 * calc_chi(x_dot_n) / (1 + torch.sqrt(1 + roughness_sq * tan2_th))


def calc_lambda_neilf(x_dot_n: Tensor, roughness: Tensor) -> Tensor:
    """ Compute lamda of GGX models.

    Refer to: https://arxiv.org/abs/2203.07182v2, Supplementary material 1.2.

    The implementation in the source code is somehow different from the above. Please refer to the following:
        https://github.com/apple/ml-neilf/blob/main/code/model/neilf_brdf.py#L158.

    Args:
        x_dot_n (Tensor): dot product of x and n (n, 1).
        roughness (Tensor): surface roughness (n, 1).

    Returns:
        (Tensor): lamda of GGX models. (n, 1)
    """

    r2 = ((1 + roughness) ** 2) / 8.
    return 0.5 / (x_dot_n * (1 - r2) + r2 + EPS)


def calc_self_masking(v_dot_n: Tensor, l_dot_n: Tensor, roughness_sq: Tensor) -> Tensor:
    """ Compute GGX self-masking term.

    Refer to: http://vclab.kaist.ac.kr/siggraphasia2018p2/polarization_supple_1.pdf.

    Args:
        v_dot_n (Tensor): (n, 1)
        l_dot_n (Tensor): (n, 1)
        roughness_sq (Tensor): (n, 1)

    Returns:
        Tensor (n, 1)
    """
    lambda_v = calc_lambda_heitz(x_dot_n=v_dot_n, roughness_sq=roughness_sq)
    lambda_l = calc_lambda_heitz(x_dot_n=l_dot_n, roughness_sq=roughness_sq)

    return lambda_v * lambda_l


def calc_self_masking_neilf(v_dot_n: Tensor, l_dot_n: Tensor, roughness: Tensor) -> Tensor:
    """ Compute GGX self-masking term.

    Refer to: https://arxiv.org/abs/2203.07182v2, Supplementary material 1.2.

    Args:
        v_dot_n (Tensor): (n, 1)
        l_dot_n (Tensor): (n, 1)
        roughness (Tensor): (n, 1)

    Returns:
        Tensor (n, 1)
    """
    lambda_v = calc_lambda_neilf(x_dot_n=v_dot_n, roughness=roughness)
    lambda_l = calc_lambda_neilf(x_dot_n=l_dot_n, roughness=roughness)

    return lambda_v * lambda_l


def calc_fresnel(v_dot_h: Tensor, f0: float = 0.95) -> Tensor:
    """ Compute GGX Fresnel term.
    Refer to: An Inexpensive BRDF Model for Physically-based Rendering [Schlick 1994].

    Args:
        v_dot_h (Tensor): (n, 1)
        f0 (float): reflection coefficient.

    Returns:
        (Tensor): calculated Fresnel term (n, 1).
    """
    return f0 + (1 - f0) * torch.pow((1 - torch.abs(v_dot_h)), 5)


def calc_fresnel_neil(v_dot_h: Tensor, base_color: Tensor, metalness: Tensor) -> Tensor:
    """ Compute GGX Fresnel term.

    Refer to: https://arxiv.org/abs/2203.07182v2, Supplementary material 1.2.

    Args:
        v_dot_h (Tensor): (n, 1)
        base_color (Tensor): (n, 3)
        metalness (Tensor): (n, 1)

    Returns:
        (Tensor): calculated Fresnel term (n, 3).
    """

    if v_dot_h.ndim != 2 or base_color.ndim != 2 or metalness.ndim != 2:
        raise ValueError("tensor's ndim must be two.")
    if v_dot_h.shape[1] != 1:
        raise ValueError("the second dimension of v_dot_h must be 1")
    if base_color.shape[1] != 3:
        raise ValueError("the second dimension of base_color must be 3")
    if metalness.shape[1] != 1:
        raise ValueError("the second dimension of metalness must be 1")

    f0 = 0.04 * (1 - metalness) + base_color * metalness
    return f0 + (1 - f0) * torch.pow(1 - v_dot_h, 5)


def calc_ggx_reflectance_baek(view_dir: Tensor, light_dir: Tensor, normal: Tensor, roughness: Tensor) -> Tensor:
    """ Compute GGX BRDF reflectance according to [Baek et al. 2018].

    Refer to: https://vclab.kaist.ac.kr/siggraphasia2018p2/polarization_author.pdf.

    Args:
        view_dir (Tensor): view directions (n, 3).
        light_dir (Tensor): light directions (n, 3).
        normal (Tensor): surface normals (n, 3).
        roughness (Tensor): (n, 1).

    Returns:
        (Tensor): GGX BRDF reflectance (n, 1).
    """

    if roughness.ndim != 2:
        raise ValueError("roughness ndim must be 2.")
    if roughness.shape[1] != 1:
        raise ValueError("the second dimension of the roughness must be 1.")

    v_dot_n = dot_product_tensor(view_dir, normal)  # (n, 1)
    l_dot_n = dot_product_tensor(light_dir, normal)  # (n, 1)

    half_vec = normalize_tensor(view_dir + light_dir)
    v_dot_h = dot_product_tensor(view_dir, half_vec)  # (n, 1)
    n_dot_h = dot_product_tensor(normal, half_vec)  # (n, 1)

    roughness_sq = roughness ** 2

    d_term = calc_normal_distribution_walter(n_dot_h=n_dot_h, roughness_sq=roughness_sq)
    g_term = calc_self_masking(v_dot_n=v_dot_n, l_dot_n=l_dot_n, roughness_sq=roughness_sq)
    f_term = calc_fresnel(v_dot_h=v_dot_h)

    return d_term * g_term * f_term / (4.0 * torch.abs(v_dot_n) * torch.abs(l_dot_n) + EPS)


def calc_ggx_reflectance_baek_no_fresnel(view_dir: Tensor,
                                         light_dir: Tensor,
                                         normal: Tensor,
                                         roughness: Tensor) -> Tensor:
    """ Compute GGX BRDF reflectance according to [Baek et al. 2018].

    Refer to: https://vclab.kaist.ac.kr/siggraphasia2018p2/polarization_author.pdf.
    Note that this function does not include the Fresnel term, because it's included in the Stokes calculation.

    Args:
        view_dir (Tensor): view directions (n, 3).
        light_dir (Tensor): light directions (n, 3).
        normal (Tensor): surface normals (n, 3).
        roughness (Tensor): (n, 1).

    Returns:
        Computed reflectance of shape (n, 1).
    """

    if roughness.ndim != 2:
        raise ValueError("roughness ndim must be 2.")
    if roughness.shape[1] != 1:
        raise ValueError("the second dimension of the roughness must be 1.")

    v_dot_n = dot_product_tensor(view_dir, normal)  # (n, 1)
    l_dot_n = dot_product_tensor(light_dir, normal)  # (n, 1)

    half_vec = normalize_tensor(view_dir + light_dir)
    n_dot_h = dot_product_tensor(normal, half_vec)  # (n, 1)

    roughness_sq = roughness ** 2

    d_term = calc_normal_distribution_walter(n_dot_h=n_dot_h, roughness_sq=roughness_sq)
    g_term = calc_self_masking(v_dot_n=v_dot_n, l_dot_n=l_dot_n, roughness_sq=roughness_sq)

    return d_term * g_term / (4.0 * torch.abs(v_dot_n) * torch.abs(l_dot_n) + EPS)


def calc_ggx_reflectance_neilf(view_dir: Tensor,
                               light_dir: Tensor,
                               normal: Tensor,
                               base_color: Tensor,
                               metalness: Tensor,
                               roughness: Tensor) -> Tensor:
    """ Compute GGX BRDF reflectance.

    Args:
        view_dir (Tensor): view directions (n, 3).
        light_dir (Tensor): light directions (n, 3).
        normal (Tensor): surface normals (n, 3).
        base_color (Tensor): surface normals (n, 3).
        metalness (Tensor): surface normals (n, 1).
        roughness (Tensor): (n, 1).

    Returns:
        (Tensor): GGX BRDF reflectance (n, 3).
    """

    if roughness.ndim != 2:
        raise ValueError("roughness ndim must be 2.")
    if roughness.shape[1] != 1:
        raise ValueError("the second dimension of the roughness must be 1.")

    half_vec = normalize_tensor(view_dir + light_dir)

    v_dot_n = dot_product_tensor(view_dir, normal)  # (n, 1)
    l_dot_n = dot_product_tensor(light_dir, normal)  # (n, 1)
    v_dot_h = dot_product_tensor(view_dir, half_vec)  # (n, 1)
    n_dot_h = dot_product_tensor(normal, half_vec)  # (n, 1)

    roughness_sq = roughness ** 2

    d_term = calc_normal_distribution_neilf(n_dot_h=n_dot_h, roughness_sq=roughness_sq)
    g_term = calc_self_masking_neilf(v_dot_n=v_dot_n, l_dot_n=l_dot_n, roughness=roughness)
    f_term = calc_fresnel_neil(v_dot_h=v_dot_h, base_color=base_color, metalness=metalness)

    return d_term * g_term * f_term
