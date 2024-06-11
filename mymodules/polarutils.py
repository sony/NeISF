# polarutils.py
""" Library for polarization-related computations.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from typing import Tuple
import math

import torch
from torch import Tensor

from .tensorutils import dot_product_tensor, normalize_tensor


AZIMUTH_MAX_RAD = torch.pi
ZENITH_MAX_RAD = torch.pi / 2.
REF_ID = 1.5
COS_BREWSTER = math.cos(math.atan(REF_ID))
EPS = 1e-07


def normalize_s0s1s2(s0: Tensor, s1: Tensor, s2: Tensor, max_val: float) -> Tuple[Tensor, Tensor, Tensor]:
    """ Normalize s0, s1, and s2 into [0, 1], [-1, 1], and [-1, 1], respectively.

    Args:
        s0 (Tensor): The first element of a Stokes vector of range [0, max_val].
        s1 (Tensor): The second element of a Stokes vector of range [0, max_val].
        s2 (Tensor): The third element of a Stokes vector of range [0, max_val].
        max_val (float): Maximum value of the s0, s1, and s2.

    Returns:
        normalized s0, s1, and s2. their ranges are [0, 1], [-1, 1], and [-1, 1], respectively.

    Note:
        This function assumes that the intensity resolution of s1 and s2 is half of s0.
        This means, actually the range of s1 and s2 after the normalization is [-0.5, 0.5].
    """

    if max_val <= 0:
        raise ValueError("max_val must be larger than zero.")
    if torch.max(s0) < 0 or torch.max(s1) < 0 or torch.max(s2) < 0:
        raise ValueError("all the values in s0, s1, and s2 must be larger than zero.")
    if max_val < torch.max(s0):
        raise ValueError("You may input wrong number: max_val < max(s0).")

    s0 = s0 / max_val
    s1 = (s1 / max_val) * 2 - 1
    s2 = (s2 / max_val) * 2 - 1

    return s0, s1, s2


def calc_s0s1s2_from_four_polar(i000: Tensor, i045: Tensor, i090: Tensor, i135: Tensor) \
        -> Tuple[Tensor, Tensor, Tensor]:
    """ Return s0, s1, and s2 from four-directional polarization images.

    Args:
        i000 (Tensor): Image captured with a linear polarizer of angle 0 [deg].
        i045 (Tensor): Image captured with a linear polarizer of angle 45 [deg].
        i090 (Tensor): Image captured with a linear polarizer of angle 90 [deg].
        i135 (Tensor): Image captured with a linear polarizer of angle 135 [deg].

    Returns:
        Computed s0, s1, and s2.
    """

    s0 = (i000 + i045 + i090 + i135) / 2.
    s1 = i000 - i090
    s2 = i045 - i135

    return s0, s1, s2


def calc_dolp_from_s0s1s2(s0: Tensor, s1: Tensor, s2: Tensor) -> Tensor:
    """  Return DoLP (Degree of Linear Polarization) from s0, s1, and s2.

    Args:
        s0 (Tensor): The first element of a Stokes vector of range [0, 1].
        s1 (Tensor): The second element of a Stokes vector of range [-1, 1].
        s2 (Tensor): The third element of a Stokes vector of range [-1, 1].

    Returns:
        Computed DoLP.

    Note:
        EPS is inside sqrt() to avoid Nan when training.
        The result is clipped to [0, 1].
    """

    dolp = torch.sqrt(s1 ** 2 + s2 ** 2 + EPS) / (s0 + EPS)
    dolp = torch.clip(dolp, 0, 1)

    return dolp


def calc_aolp_from_s1s2(s1: Tensor, s2: Tensor) -> Tensor:
    """ Return AoLP (Angle of Linear Polarization) from s1 and s2.

    Args:
        s1 (Tensor): The second element of a Stokes vector of range [-1, 1].
        s2 (Tensor): The third element of a Stokes vector of range [-1, 1].

    Returns:
        Computed AoLP (rad) of range [0, pi).
    """

    aolp = torch.atan2(s2, s1 + EPS)
    mask = (aolp < 0)
    aolp[mask] = aolp[mask] + AZIMUTH_MAX_RAD * 2  # polarization ambiguity
    aolp /= 2.
    aolp = torch.clip(aolp, 0, AZIMUTH_MAX_RAD)

    return aolp


def calc_s1s2_from_s0_dolp_aolp(s0: Tensor, dolp: Tensor, aolp: Tensor) -> Tuple[Tensor, Tensor]:
    """ Compute s1 and s2 from s0, DoLP, and AoLP.

    Args:
        s0 (Tensor): The first element of a Stokes vector of range [0, 1].
        dolp (Tensor): DoLP [0, 1].
        aolp (Tensor): AoLP (rad) of range [0, pi).

    Returns:
        Computed s1 and s2.
    """

    s1 = s0 * dolp * torch.cos(2 * aolp)
    s2 = s0 * dolp * torch.sin(2 * aolp)

    return s1, s2


def get_beta_specular(theta_rad: Tensor, ref_id: float = REF_ID) -> Tensor:
    """ Compute beta of specular reflection.

    Refer to: Eq. 4 in https://arxiv.org/pdf/2203.13458.pdf.

    Args:
        theta_rad (Tensor): Zenith angle [rad].
        ref_id (float): Refractive index.

    Returns:
        Calculated beta.
    """

    sin_t, cos_t = torch.sin(theta_rad), torch.cos(theta_rad)
    sin_t2 = sin_t * sin_t

    beta_s_num = 2 * sin_t2 * cos_t * torch.sqrt(ref_id ** 2 - sin_t2 + EPS)
    beta_s_den = ref_id ** 2 - sin_t2 - sin_t2 * ref_id ** 2 + 2 * sin_t2 * sin_t2

    beta_s = beta_s_num / (beta_s_den + EPS)  # sometimes there are tiny values smaller than zero.
    beta_s = torch.clip(beta_s, 0., 1.)

    return beta_s


def get_beta_diffuse(theta_rad: Tensor, ref_id: float = REF_ID) -> Tensor:
    """ Compute beta of diffuse reflection.

    Refer to: Eq. 4 in https://arxiv.org/pdf/2203.13458.pdf.

    Args:
        theta_rad (Tensor): Zenith angle [rad].
        ref_id (float): Refractive index.

    Returns:
        Calculated beta.
    """

    sin_t, cos_t = torch.sin(theta_rad), torch.cos(theta_rad)
    sin_t2 = sin_t * sin_t

    beta_d_num = sin_t2 * (ref_id - 1 / ref_id) ** 2
    beta_d_den = \
        2 + 2 * ref_id ** 2 - sin_t2 * (ref_id + 1 / ref_id) ** 2 + 4 * cos_t * torch.sqrt(ref_id ** 2 - sin_t2 + EPS)

    beta_d = beta_d_num / (beta_d_den + EPS)

    return beta_d


def calc_diffuse_zenith_from_dolp(dolp: Tensor, ref_id: float = REF_ID) -> Tensor:
    """ Compute zenith angles from DoLP.

    Args:
        dolp(Tensor): DoLP.
        ref_id (float): Refractive index.

    Returns:
        Computed zenith angles [rad].
    """

    max_diffuse_dolp = get_beta_diffuse(theta_rad=torch.tensor(ZENITH_MAX_RAD, device=dolp.device))
    dolp = torch.clamp(dolp, min=torch.tensor(0., device=dolp.device), max=max_diffuse_dolp)

    val_a = 2 * (1 - dolp) - (1 + dolp) * (ref_id ** 2 + 1 / (ref_id ** 2))
    val_b = 4 * dolp
    val_c = 1 + ref_id ** 2
    val_d = 1 - ref_id ** 2

    sin_val_sqrt_num = -1 * val_b * (val_c * (val_a + val_b) - torch.sqrt(
        val_c ** 2 * (val_a + val_b) ** 2 - val_d ** 2 * (val_a ** 2 - val_b ** 2) + EPS))
    sin_val_sqrt_denum = 2 * (val_a ** 2 - val_b ** 2)

    sin_val = sin_val_sqrt_num / (sin_val_sqrt_denum + EPS)

    cd = torch.sqrt(sin_val + EPS)
    cd_clip = torch.clip(cd, -1 + EPS, 1 - EPS)  # this is necessary to prevent the gradient from becoming inf.

    out_zenith = torch.asin(cd_clip)
    out_zenith = torch.clip(out_zenith, 0, ZENITH_MAX_RAD)

    return out_zenith


def calc_ts_tp(cos_theta: Tensor, ref_id: float = REF_ID) -> Tuple[Tensor, Tensor]:
    """ Compute T_s and T_p of diffuse Fresnel term.

    Refer to: https://vclab.kaist.ac.kr/siggraphasia2018p2/polarization_author.pdf.

    Args:
        cos_theta (Tensor): Cos values of the zenith angles.
        ref_id (float): Refractive index.

    Returns:
        Computed T_s, T_p.
    """

    eta_sq = ref_id ** 2
    sin_theta_sq = 1 - cos_theta ** 2
    sqr = torch.sqrt(eta_sq - sin_theta_sq + EPS)

    t_s = 4 * cos_theta * sqr / ((cos_theta + sqr) ** 2 + EPS)
    t_p = 4 * cos_theta * sqr * eta_sq / ((eta_sq * cos_theta + sqr) ** 2 + EPS)

    return t_s, t_p


def calc_rs_rp(cos_theta: Tensor, ref_id: float = REF_ID) -> Tuple[Tensor, Tensor]:
    """ Compute R_s and R_p of diffuse Fresnel term.

    Refer to: https://vclab.kaist.ac.kr/siggraphasia2018p2/polarization_author.pdf.

    Args:
        cos_theta (Tensor): Cos values of the zenith angles.
        ref_id (float): Refractive index.

    Returns:
        Computed T_s, T_p, R_s, and R_p.
    """

    eta_sq = ref_id ** 2
    sin_theta_sq = 1 - cos_theta ** 2
    sqr = torch.sqrt(eta_sq - sin_theta_sq + EPS)

    r_s = ((cos_theta - sqr) / (cos_theta + sqr)) ** 2
    r_p = ((eta_sq * cos_theta - sqr) / (eta_sq * cos_theta + sqr)) ** 2

    return r_s, r_p


def calc_normal_from_dolp_and_aolp(dolp: Tensor, aolp: Tensor, ref_id: float = REF_ID) -> Tensor:
    """ Compute surface normal based on the diffuse polarization model.

    Args:
        dolp (Tensor): DoLP of shape (n, 1).
        aolp (Tensor): AoLP [rad] of shape (n, 1).
        ref_id (float): Refractive index.

    Returns:
        Computed surface normal of shape (n, 3).
    """

    if aolp.dim() != 2 or dolp.dim() != 2:
        raise ValueError("The dimension of input must be 2.")
    if aolp.shape[1] != 1 or dolp.shape[1] != 1:
        raise ValueError("The second dimension of input must be 1.")

    diffuse_zenith = calc_diffuse_zenith_from_dolp(dolp, ref_id)

    normal_x = torch.cos(aolp) * torch.sin(diffuse_zenith)
    normal_y = torch.sin(aolp) * torch.sin(diffuse_zenith)
    normal_z = torch.cos(diffuse_zenith)

    out_normal = torch.cat([normal_x, normal_y, normal_z], dim=1)

    return out_normal


def rotate_normal_from_world_to_tangent(w2c: Tensor, rays_d: Tensor, normals: Tensor) -> Tensor:
    """ Rotate surface normals from world coordinate to tangent coordinate.

    Args:
          w2c (Tensor): world to camera matrix (3, 3) or (n, 3, 3).
          rays_d (Tensor): ray directions in a world coordinate (n, 3).
          normals (Tensor): surface normals represented in a world coordinate (n, 3).

    Returns:
        surface normals rotated to tangent coordinate (n, 3).
    """

    if rays_d.ndim != 2 or normals.ndim != 2:
        raise ValueError("ndim of ray_d and normals must be two.")
    if rays_d.shape[1] != 3 or normals.shape[1] != 3:
        raise ValueError("the second dimension of the rays_d and normals must be three.")
    if rays_d.shape != normals.shape:
        raise ValueError("rays_d and normals must be the same shape.")

    if w2c.ndim == 2:  # all the rays share the same matrix (3, 3).
        cam_x = w2c[0:1, :]  # (1, 3).
    elif w2c.ndim == 3:
        cam_x = w2c[:, 0, :]  # (n, 3).
    else:
        raise ValueError("w2c shape is wrong.")

    tan_y = normalize_tensor(torch.cross(cam_x, rays_d))  # (n, 3)
    tan_x = normalize_tensor(torch.cross(rays_d, tan_y))  # (n, 3)

    mat_w2t = torch.cat((tan_x.unsqueeze(-1), tan_y.unsqueeze(-1), - rays_d.unsqueeze(-1)), dim=-1)

    return torch.bmm(normals.unsqueeze(1), mat_w2t)[:, 0, :]


def rotate_normal_from_tangent_to_world(w2c: Tensor, rays_d: Tensor, normal_tan: Tensor) -> Tensor:
    """ Rotate surface normals from world coordinate to tangent coordinate.

    Args:
          w2c (Tensor): world to camera matrix (3, 3).
          rays_d (Tensor): ray directions in a world coordinate (n, 3).
          normal_tan (Tensor): surface normals represented in a tangent coordinate (n, 3).

    Returns:
        Surface normal rotated to world coordinate (n, 3).
    """

    if rays_d.ndim != 2 or normal_tan.ndim != 2:
        raise ValueError("ndim of ray_d and normals must be two.")
    if rays_d.shape[1] != 3 or normal_tan.shape[1] != 3:
        raise ValueError("the second dimension of the rays_d and normals must be three.")
    if rays_d.shape != normal_tan.shape:
        raise ValueError("rays_d and normals must be the same shape.")

    cam_x = w2c[0:1, :]  # (1, 3).

    tan_y = normalize_tensor(torch.cross(cam_x, rays_d))  # (n, 3)
    tan_x = normalize_tensor(torch.cross(rays_d, tan_y))  # (n, 3)

    mat_t2w = torch.cat((tan_x.unsqueeze(1), tan_y.unsqueeze(1), - rays_d.unsqueeze(1)), dim=1)

    return torch.bmm(normal_tan.unsqueeze(1), mat_t2w)[:, 0, :]


def calc_diffuse_stokes(wo_wld: Tensor, wi_wld: Tensor, n_wld: Tensor, w2c: Tensor, ref_id: float = REF_ID) -> Tensor:
    """ Compute diffuse Stokes vectors according to the view and light directions.

    Args:
        wo_wld (Tensor): view directions in world coordinate (n, 3).
        wi_wld (Tensor): light directions in world coordinate (n, 3).
        n_wld (Tensor): surface normals in world coordinate (n, 3).
        w2c (Tensor): world to camera matrix (3, 3).
        ref_id (float): refractive index.

    Returns:
        Computed diffuse Stokes vectors (n, 3).

    Note:
        - This function assumes the incoming light is not polarized.
        - This function is almost the same as `calc_diffuse_stokes_mitsuba()`, but sometimes different for some reasons.
    """

    # calculate theta_i and theta_o.
    cos_i = dot_product_tensor(wi_wld, n_wld)
    cos_o = dot_product_tensor(wo_wld, n_wld)

    # calculate phi_o.
    n_cam = n_wld @ w2c.T  # (n, 3)
    wo_cam = wo_wld @ w2c.T  # (n, 3)
    forward_cam = torch.tensor([[0., 0., 1.]], device=n_cam.device)  # (1, 3)

    d = normalize_tensor(torch.cross(torch.cross(-wo_cam, n_cam), forward_cam))  # (n, 3)

    cos_phi = d[:, 1:2]
    sin_phi = d[:, 0:1]
    sin_2phi = 2 * sin_phi * cos_phi
    cos_2phi = 1 - 2 * sin_phi * sin_phi

    ts_i, tp_i = calc_ts_tp(cos_theta=cos_i, ref_id=ref_id)
    ts_o, tp_o = calc_ts_tp(cos_theta=cos_o, ref_id=ref_id)

    t_plus_i = (ts_i + tp_i) / 2.
    t_plus_o = (ts_o + tp_o) / 2.
    t_minus_o = (ts_o - tp_o) / 2.

    diffuse_stokes = torch.zeros_like(wo_wld)  # (n, 3)
    diffuse_stokes[:, 0:1] = t_plus_o * t_plus_i
    diffuse_stokes[:, 1:2] = t_minus_o * t_plus_i * cos_2phi
    diffuse_stokes[:, 2:] = - t_minus_o * t_plus_i * sin_2phi

    return diffuse_stokes


def calc_diffuse_stokes_mitsuba(wo_wld: Tensor, wi_wld: Tensor, n_wld: Tensor, w2c: Tensor, ref_id: float = REF_ID) \
        -> Tensor:
    """ Computes diffuse Stokes vectors according to the view and light directions.

    Args:
        wo_wld (Tensor): view directions in world coordinate (n, 3).
        wi_wld (Tensor): light directions in world coordinate (n, 3).
        n_wld (Tensor): surface normals in world coordinate (n, 3).
        w2c (Tensor): world to camera matrix (3, 3).
        ref_id (float): refractive index.

    Returns:
        (Tensor): diffuse Stokes vectors (n, 3).

    Note:
        - This function assumes the incoming light is not polarized.
        - Unlike `calc_diffuse_stokes()`, this is confirmed to produce the same value as Mitsuba.
    """

    # calculate theta_i and theta_o.
    cos_i = dot_product_tensor(wi_wld, n_wld)
    cos_o = dot_product_tensor(wo_wld, n_wld)

    c2w = w2c.T

    # calculate phi_o.
    # refer to https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html
    forward_wld = torch.tensor([[0., 1., 0.]],
                               device=n_wld.device) @ c2w.T  # y-axis of the camera in the world coordinate (1, 3)
    current_basis = normalize_tensor(torch.cross(n_wld, wo_wld))  # stokes basis of the BRDF
    target_basis = normalize_tensor(torch.cross(wo_wld, forward_wld))  # stokes basis of the camera
    cos_phi = dot_product_tensor(current_basis, target_basis)  # rotation angle

    # flip phi according to the forward ray direction
    mask = dot_product_tensor(wo_wld, torch.cross(current_basis, target_basis)) < 0.0
    flip = mask * (-2.) + 1.
    sin_phi = torch.sqrt(1. - cos_phi * cos_phi + EPS) * flip

    sin_2phi = 2. * sin_phi * cos_phi
    cos_2phi = 1. - 2. * sin_phi * sin_phi

    ts_i, tp_i = calc_ts_tp(cos_theta=cos_i, ref_id=ref_id)
    ts_o, tp_o = calc_ts_tp(cos_theta=cos_o, ref_id=ref_id)

    t_plus_i = (ts_i + tp_i) / 2.
    t_plus_o = (ts_o + tp_o) / 2.
    t_minus_o = (ts_o - tp_o) / 2.

    diffuse_stokes = torch.zeros_like(wo_wld)  # (n, 3)
    diffuse_stokes[:, 0:1] = t_plus_o * t_plus_i
    diffuse_stokes[:, 1:2] = t_minus_o * t_plus_i * cos_2phi
    diffuse_stokes[:, 2:] = - t_minus_o * t_plus_i * sin_2phi

    return diffuse_stokes


def calc_diffuse_stokes_no_wi(wo_wld: Tensor, n_wld: Tensor, w2c: Tensor, ref_id: float = REF_ID) \
        -> Tuple[Tensor, Tensor]:
    """ Compute diffuse Stokes vectors only from viewing angles.

    Refer to: https://arxiv.org/pdf/2203.13458.pdf Eq. 4.

    Args:
        wo_wld (Tensor): view directions in world coordinate (n, 3).
        n_wld (Tensor): surface normals in world coordinate (n, 3).
        w2c (Tensor): world to camera matrix (3, 3).
        ref_id (float): refractive index.

    Returns:
        Computed diffuse s1 and s2 of shape (n, 1), (n, 1).
    """

    # calculate theta_o.
    cos_o = dot_product_tensor(wo_wld, n_wld)

    # calculate phi_o.
    n_cam = n_wld @ w2c.T  # (n, 3)
    wo_cam = wo_wld @ w2c.T  # (n, 3)
    forward_cam = torch.tensor([[0., 0., 1.]], device=n_cam.device)  # (1, 3)

    d = normalize_tensor(torch.cross(torch.cross(-wo_cam, n_cam), forward_cam))  # (n, 3)

    cos_phi = d[:, 1:2]
    sin_phi = d[:, 0:1]
    sin_2phi = 2 * sin_phi * cos_phi
    cos_2phi = 1 - 2 * sin_phi * sin_phi

    ts_o, tp_o = calc_ts_tp(cos_theta=cos_o, ref_id=ref_id)
    beta_d = (ts_o - tp_o) / (ts_o + tp_o + EPS)

    s1 = beta_d * cos_2phi
    s2 = - beta_d * sin_2phi

    return s1, s2


def calc_diffuse_fresnel(wo_wld: torch.Tensor,
                         wi_wld: torch.Tensor,
                         n_wld: torch.Tensor,
                         ref_id: float = REF_ID) -> torch.Tensor:
    """ Compute the unpolarized fresnel calculation of the diffuse term (refraction)

    Args:
        wo_wld (torch.Tensor): view directions in world coordinate (n, 3).
        wi_wld (torch.Tensor): light directions in world coordinate (n, 3).
        n_wld (torch.Tensor): surface normals in world coordinate (n, 3).
        ref_id (float): refraction index.

    Returns:
        (torch.Tensor): the value of fresnel term for the two times refraction (n, 1).
    """

    cos_i = dot_product_tensor(wi_wld, n_wld)
    cos_o = dot_product_tensor(wo_wld, n_wld)

    ts_i, tp_i = calc_ts_tp(cos_theta=cos_i, ref_id=ref_id)
    ts_o, tp_o = calc_ts_tp(cos_theta=cos_o, ref_id=ref_id)

    t_plus_i = (ts_i + tp_i) / 2.
    t_plus_o = (ts_o + tp_o) / 2.

    return t_plus_o * t_plus_i


def calc_diffuse_stokes_full(wo_wld: Tensor,
                             wi_wld: Tensor,
                             n_wld: Tensor,
                             cam_axis: Tensor,
                             s0: Tensor,
                             s1: Tensor,
                             ref_id: float = REF_ID) -> Tuple[Tensor, Tensor, Tensor]:
    """ Compute diffuse Stokes vectors according to the view and light directions.
    Unlike `calc_diffuse_stokes()` and `calc_diffuse_stokes_mitsuba()`, this function handles polarized illumination.

    Args:
        wo_wld (Tensor): view directions in world coordinate (n, 3).
        wi_wld (Tensor): light directions in world coordinate (n, 3).
        n_wld (Tensor): surface normals in world coordinate (n, 3).
        cam_axis (Tensor): camera axis (pointing right) in world coordinate (n, 3).
        s0 (Tensor): the first element of Stokes vector (n, 3).
        s1 (Tensor): the second element of Stokes vector (n, 3).
        ref_id (float): refractive index.

    Returns:
        (Tensor): the first element of the outgoing Stokes vectors (n, 3)
        (Tensor): the second element of the outgoing Stokes vectors (n, 3)
        (Tensor): the third element of the outgoing Stokes vectors (n, 3)

    Note:
        We assume the incident Stokes vectors are already rotated.
        Namely, the incident Stokes vectors already share the same coordinate frames with the surface.
    """

    # calculate theta_i and theta_o.
    cos_i = dot_product_tensor(wi_wld, n_wld)
    cos_o = dot_product_tensor(wo_wld, n_wld)

    # calculate phi_o.
    # refer to https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html
    current_basis = normalize_tensor(torch.cross(n_wld, wo_wld))  # stokes basis of the BRDF
    target_basis = normalize_tensor(torch.cross(wo_wld, cam_axis))  # stokes basis of the camera
    cos_phi = dot_product_tensor(current_basis, target_basis)  # rotation angle

    # flip phi according to the forward ray direction
    mask = dot_product_tensor(wo_wld, torch.cross(current_basis, target_basis)) < 0.0
    flip = mask * (-2.) + 1.
    sin_phi = torch.sqrt(1. - cos_phi * cos_phi + EPS) * flip

    sin_2phi = 2. * sin_phi * cos_phi
    cos_2phi = 1. - 2. * sin_phi * sin_phi

    # calculate Ts and Tp
    ts_i, tp_i = calc_ts_tp(cos_theta=cos_i, ref_id=ref_id)
    ts_o, tp_o = calc_ts_tp(cos_theta=cos_o, ref_id=ref_id)

    t_plus_i = (ts_i + tp_i) / 2.
    t_minus_i = (ts_i - tp_i) / 2.
    t_plus_o = (ts_o + tp_o) / 2.
    t_minus_o = (ts_o - tp_o) / 2.

    diffuse_stokes_s0 = (s0 * t_plus_o * t_plus_i) + (s1 * t_plus_o * t_minus_i)
    diffuse_stokes_s1 = (s0 * t_minus_o * t_plus_i * cos_2phi) + (s1 * t_minus_o * t_minus_i * cos_2phi)
    diffuse_stokes_s2 = (-s0 * t_minus_o * t_plus_i * sin_2phi) + (-s1 * t_minus_o * t_minus_i * sin_2phi)

    return diffuse_stokes_s0, diffuse_stokes_s1, diffuse_stokes_s2


def calc_specular_fresnel(wo_wld: torch.Tensor,
                          wi_wld: torch.Tensor,
                          ref_id: float = REF_ID) -> torch.Tensor:
    """ Compute the unpolarized fresnel calculation of the specular term.

    Args:
        wo_wld (torch.Tensor): view directions in world coordinate (n, 3).
        wi_wld (torch.Tensor): light directions in world coordinate (n, 3).
        ref_id (float): refraction index.

    Returns:
        (torch.Tensor): the value of fresnel term for the specular reflection (n, 1).
    """

    hvec_wld = normalize_tensor(wo_wld + wi_wld)

    cos_theta = dot_product_tensor(wo_wld, hvec_wld)  # (n, 1).

    r_s, r_p = calc_rs_rp(cos_theta, ref_id=ref_id)  # (n, 1), (n, 1).
    r_plus = (r_s + r_p) / 2.

    return r_plus


def calc_specular_stokes(wo_wld: Tensor, wi_wld: Tensor, w2c: Tensor, ref_id: float = REF_ID) -> Tensor:
    """ Compute specular Stokes vectors according to the view and light directions.

    Args:
        wo_wld (Tensor): view directions in world coordinate (n, 3).
        wi_wld (Tensor): light directions in world coordinate (n, 3).
        w2c (Tensor): world to camera matrix (3, 3).
        ref_id (float): refractive index.

    Returns:
        Computed specular Stokes vectors (n, 3).

    Note:
        - This function assumes the incoming light is not polarized.
        - This function is almost the same as `calc_specular_stokes_mitsuba()`, but sometimes different for some reasons.
    """

    hvec_wld = normalize_tensor(wo_wld + wi_wld)  # half vector (n, 3).

    # calculate phi_o
    hvec_cam = hvec_wld @ w2c.T
    wo_cam = wo_wld @ w2c.T  # (n, 3)
    forward_cam = torch.tensor([[0., 0., 1.]], device=hvec_cam.device)  # (1, 3)

    d = normalize_tensor(torch.cross(torch.cross(-wo_cam, hvec_cam), forward_cam))  # (n, 3)

    cos_phi = d[:, 1:2]
    sin_phi = d[:, 0:1]
    sin_2phi = 2 * sin_phi * cos_phi
    cos_2phi = 1 - 2 * sin_phi * sin_phi

    # calculate Rs and Rp
    cos_theta = dot_product_tensor(wo_wld, hvec_wld)  # (n, 1).
    r_s, r_p = calc_rs_rp(cos_theta, ref_id=ref_id)  # (n, 1), (n, 1).

    r_plus = (r_s + r_p) / 2.
    r_minus = (r_s - r_p) / 2.

    specular_stokes = torch.zeros_like(wo_wld)  # (n, 3).
    specular_stokes[:, 0:1] = r_plus
    specular_stokes[:, 1:2] = r_minus * cos_2phi
    specular_stokes[:, 2:] = - r_minus * sin_2phi

    return specular_stokes


def calc_specular_stokes_mitsuba(wo_wld: Tensor, wi_wld: Tensor, w2c: Tensor, ref_id: float = REF_ID) -> Tensor:
    """ Compute specular Stokes vectors according to the view and light directions.

    Args:
        wo_wld (Tensor): view directions in world coordinate (n, 3).
        wi_wld (Tensor): light directions in world coordinate (n, 3).
        w2c (Tensor): world to camera matrix (3, 3).
        ref_id (float): refractive index.

    Returns:
        (Tensor): specular Stokes vectors (n, 3).

    Note:
        - This function assumes the incoming light is not polarized.
        - Unlike `calc_specular_stokes()`, this is confirmed to produce the same value as Mitsuba.
    """

    hvec_wld = normalize_tensor(wo_wld + wi_wld)  # half vector (n, 3).

    c2w = w2c.T  # camera to world matrix

    # calculate phi_o
    # refer to https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html
    forward_wld = torch.tensor([[0., 1., 0.]],
                               device=hvec_wld.device) @ c2w.T  # y-axis of the camera in the world coordinate(1, 3)
    current_basis = normalize_tensor(torch.cross(hvec_wld, wo_wld))  # stokes basis of the BRDF
    target_basis = normalize_tensor(torch.cross(wo_wld, forward_wld))  # stokes basis of the camera
    cos_phi = dot_product_tensor(current_basis, target_basis)  # rotation angle

    # flip phi according to the forward direction
    mask = dot_product_tensor(wo_wld, torch.cross(current_basis, target_basis)) < 0.0
    flip = mask * (-2.) + 1.
    sin_phi = torch.sqrt(1. - cos_phi * cos_phi + EPS) * flip

    sin_2phi = 2. * sin_phi * cos_phi
    cos_2phi = 1. - 2. * sin_phi * sin_phi

    # calculate Rs and Rp
    cos_theta = dot_product_tensor(wo_wld, hvec_wld)  # (n, 1).
    r_s, r_p = calc_rs_rp(cos_theta, ref_id=ref_id)  # (n, 1), (n, 1).

    r_plus = (r_s + r_p) / 2.
    r_minus = (r_s - r_p) / 2.

    specular_stokes = torch.zeros_like(wo_wld)  # (n, 3).
    specular_stokes[:, 0:1] = r_plus
    specular_stokes[:, 1:2] = r_minus * cos_2phi
    specular_stokes[:, 2:] = - r_minus * sin_2phi

    return specular_stokes


def calc_specular_stokes_full(wo_wld: Tensor,
                              wi_wld: torch.Tensor,
                              cam_axis: Tensor,
                              s0: Tensor,
                              s1: Tensor,
                              s2: Tensor,
                              ref_id: float = REF_ID) -> Tuple[Tensor, Tensor, Tensor]:
    """ Compute specular Stokes vectors according to the view and light directions.
    Unlike `calc_specular_stokes()` and `calc_specular_stokes_mitsuba()`, this function handles polarized illumination.

    Args:
        wo_wld (torch.Tensor): view directions in world coordinate (n, 3).
        wi_wld (torch.Tensor): incident ray direction in world coordinate (n, 3).
        cam_axis (Tensor): camera axis (pointing right) in world coordinate (n, 3).
        s0 (torch.Tensor): the first element (intensity) of the incident Stokes vectors (n, 3).
        s1 (torch.Tensor): the second element of the incident Stokes vectors (n, 3).
        s2 (torch.Tensor): the third element of the incident Stokes vectors (n, 3).
        ref_id (float): refractive index.

    Returns:
        (Tensor): the first element of the outgoing Stokes vectors (n, 3)
        (Tensor): the second element of the outgoing Stokes vectors (n, 3)
        (Tensor): the third element of the outgoing Stokes vectors (n, 3)

    Note:
        We assume the incident Stokes vectors are already rotated.
        Namely, the incident Stokes vectors already share the same coordinate frames with the surface.
    """

    hvec_wld = normalize_tensor(wo_wld + wi_wld)  # half vector (n, 3).

    # calculate phi_o.
    # refer to https://mitsuba.readthedocs.io/en/stable/src/key_topics/polarization.html
    current_basis = normalize_tensor(torch.cross(hvec_wld, wo_wld))  # stokes basis of the BRDF
    target_basis = normalize_tensor(torch.cross(wo_wld, cam_axis))  # stokes basis of the camera
    cos_phi = dot_product_tensor(current_basis, target_basis)  # rotation angle

    # flip phi according to the forward direction
    mask = dot_product_tensor(wo_wld, torch.cross(current_basis, target_basis)) < 0.0
    flip = mask * (-2.) + 1.
    sin_phi = torch.sqrt(1. - cos_phi * cos_phi + EPS) * flip

    sin_2phi = 2. * sin_phi * cos_phi
    cos_2phi = 1. - 2. * sin_phi * sin_phi

    # calculate Rs and Rp
    cos_theta = dot_product_tensor(wo_wld, hvec_wld)  # (n, 1).
    r_s, r_p = calc_rs_rp(cos_theta, ref_id=ref_id)  # (n, 1), (n, 1).

    r_plus = (r_s + r_p) / 2.
    r_minus = (r_s - r_p) / 2.
    r_cross = torch.sqrt(r_s * r_p + EPS)

    # cos_delta = -1 or 1 when the incident angle < or > the Brewster angle
    cos_delta = (cos_theta < COS_BREWSTER) * 2. - 1.

    specular_stokes_s0 = (s0 * r_plus) + (s1 * r_minus)
    specular_stokes_s1 = (s0 * r_minus * cos_2phi) + (s1 * r_plus * cos_2phi) + (s2 * r_cross * sin_2phi * cos_delta)
    specular_stokes_s2 = (-s0 * r_minus * sin_2phi) + (-s1 * r_plus * sin_2phi) + (s2 * r_cross * cos_2phi * cos_delta)

    return specular_stokes_s0, specular_stokes_s1, specular_stokes_s2
