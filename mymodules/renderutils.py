# renderutils.py
""" Library for rendering (e.g. camera poses, casting rays, volume rendering, and ray tracing).

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from mymodules.globals import BOUNDING_SPHERE_R
from mymodules.models.volsdf import SDFNet
from mymodules.samplers.utils import get_sphere_intersections
from mymodules.tensorutils import normalize_tensor, dot_product_tensor, calc_rot_mat_from_two_vec

ILLUM_AREA = torch.pi * 2.  # solid angle of the fibonacci sampling


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    """ this function normalizes 1D vector.

    Args:
        v (np.ndarray): 1D vector (3,).

    Returns:
        normalized 1D vector (3,).
    """

    if v.ndim != 1 or v.shape[0] != 3:
        raise ValueError("input vector's shape must be (3,)")
    return v / np.linalg.norm(v)


def get_hwf_from_poses_bounds(poses: np.ndarray) -> Tuple[int, int, float]:
    """ this function extract the height, width, and focal length from `poses_bounds.npy`.

    Args:
        poses (np.ndarray): poses_bounds of shape (n_images, 3, 5).

    Returns:
        (int) height of images.
        (int) width of images.
        (float) focal length of images.
    """
    if poses.ndim != 3:
        raise ValueError("the shape of arr is wrong.")
    if poses.shape[1] != 3 or poses.shape[2] != 5:
        raise ValueError("the second and third axis must be 3 and 5, respectively.")

    hwf = poses[:, :, -1]  # (n_images, 3)

    height_all = hwf[:, 0]  # (n_images,)
    width_all = hwf[:, 1]  # (n_images,)
    focal_all = hwf[:, 2]  # (n_images,)

    if not np.all(height_all[0] == height_all):
        raise ValueError("height must be the same for all the frames.")
    if not np.all(width_all[0] == width_all):
        raise ValueError("width must be the same for all the frames.")
    if not np.all(focal_all[0] == focal_all):
        raise ValueError("focal length must be the same for all the frames.")

    return int(height_all[0]), int(width_all[0]), focal_all[0]


def get_center_point(poses: np.ndarray) -> np.ndarray:
    """ This function computes the intersection of all cameras based on least square.
    Refer to: https://github.com/lioryariv/volsdf/blob/main/data/preprocess/normalize_cameras.py

    Args:
        poses (np.ndarray): camera poses of shape (n_images, 3, 4).

    Returns:
        computed intersections of all cameras view direction (3,).
    """

    n_images = poses.shape[0]

    a_mat = np.zeros((3 * n_images, 3 + n_images))
    b = np.zeros((3 * n_images, 1))

    for i in range(n_images):
        poses_sub = poses[i]  # (3, 4)
        r_mat = poses_sub[:, :3]  # (3, 3)
        c = poses_sub[:, 3:]  # (3, 1)

        v = r_mat[:, 2]

        a_mat[3 * i:(3 * i + 3), :3] = np.eye(3)
        a_mat[3 * i:(3 * i + 3), 3 + i] = -v
        b[3 * i:(3 * i + 3)] = c

    soll = np.linalg.pinv(a_mat) @ b  # (n_images, 1)

    return soll[:3].flatten()


def get_average_poses(poses: np.ndarray) -> np.ndarray:
    """ This function aligns the average y-axis of all cameras to the global z-axis.

    Args:
        poses (np.ndarray): camera poses of shape (n_images, 3, 4).

    Return:
        rotation matrix for normalizing all the cameras (3, 4).

    Note:
        average value of the y-axis of all the cameras -> global y-axis.
        (average value of the y-axis of all the cameras).cross(y_axis) -> global z_axis.
        y_axis.cross(z_axis) -> global x_axis.
    """

    result = np.zeros((3, 4), dtype=np.float32)

    avg_y = _normalize_vector(poses[..., 1].mean(0))  # (3,)

    if avg_y[2] > 0.98:  # already cameras are aligned. return identity matrix.
        result[:, :3] = np.eye(3)
        print("Camera coordinate already normalized. No rotation.")
    else:
        y_axes = avg_y
        avg_x = _normalize_vector(poses[..., 0].mean(0))  # (3,)
        z_axes = _normalize_vector(np.cross(avg_x, y_axes))
        x_axes = _normalize_vector(np.cross(y_axes, z_axes))

        result[:, 0] = x_axes
        result[:, 1] = -z_axes
        result[:, 2] = y_axes

    return result


def normalize_all_cameras(poses: np.ndarray) -> np.ndarray:
    """ this function locates the target object to the global origin, to set the average value of camera's y_axis to
    global y_axis, and locates all the cameras within a sphere of radius=3.

    Args:
        poses (np.ndarray): camera poses of shape (n_images, 3, 4).

    Returns:
        normalized cameras (np.ndarray).
    """

    center = get_center_point(poses)
    pose_avg = get_average_poses(poses)

    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3, :] = pose_avg
    pose_avg_homo[:3, 3] = center

    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (n_images, 1, 4)
    poses_homo = np.concatenate([poses, last_row], 1)  # (n_images, 4, 4)

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (n_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (n_images, 3, 4)

    # Normalize the scale to locate all the cameras within a sphere of radius 3.
    # Refer to: https://arxiv.org/abs/2106.12052) B.1.
    cam_loc_max = np.max(np.linalg.norm(poses_centered[:, :, 3], axis=1))
    poses_centered[..., 3] *= BOUNDING_SPHERE_R / cam_loc_max / 1.1

    return poses_centered


def generate_rays_cam_coord(height: int, width: int, focal: float) -> Tensor:
    """ Generate ray directions for all pixels in CAMERA COORDINATE.

    Args:
        height (int): height of image.
        width (int): width of image.
        focal (float): focal length of image.

    Returns:
        (Tensor): generated rays in camera coordinate.
    """

    if focal == 0:
        raise ValueError("Your focal length is zero!")
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("height and width must be even numbers.")

    j, i = torch.meshgrid(torch.arange(0, height, 1), torch.arange(0, width, 1), indexing="ij")

    cx = width / 2.
    cy = height / 2.

    ray_directions = torch.stack([(i - cx) / focal, -(j - cy) / focal, -torch.ones_like(i)], -1)  # right, up, back!

    return ray_directions


def rotate_rays_to_world_coord(rays_cam_coord: Tensor, c2w: Tensor) -> Tuple[Tensor, Tensor]:
    """ Get ray origin and normalized directions in WORLD COORDINATE.

    Args:
        rays_cam_coord (Tensor): ray directions in camera coordinate. (height, width, 3).
        c2w (Tensor): camera to world matrices (3, 4).

    Returns:
        origins and directions of rays.
    """

    rays_d = rays_cam_coord @ c2w[:, :3].T  # (h, w, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    rays_o = c2w[:, 3].expand(rays_d.shape)  # (h, w, 3)

    rays_d = rays_d.reshape(-1, 3)  # (h * w, 3)
    rays_o = rays_o.reshape(-1, 3)

    return rays_o, rays_d


def volume_rendering(z_vals: Tensor, sdf_net: SDFNet, sdf_val: Tensor, device: torch.device) -> Tensor:
    """ This function computes weights for each position.
    Refer to: https://github.com/lioryariv/volsdf/blob/main/code/model/network.py

    Args:
        z_vals (Tensor): sampled z according to the volsdf (b_size, zn_samples).
        sdf_net (SDFNet): sdf-net.
        sdf_val (Tensor): sdf values of the sampled position (b_size * zn_samples, 1).
        device (torch.device): the device you are using.

    Returns:
        (Tensor): calculated weight.
    """

    density_flat = sdf_net.density(sdf_val)
    density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size, zn_samples)

    dists = z_vals[:, 1:] - z_vals[:, :-1]  # (batch_size, zn_samples - 1)
    dists = torch.cat([dists, torch.tensor([1e10], device=device).unsqueeze(0).repeat(dists.shape[0], 1)], -1)
    # here we add a quite large value at the end to make sure all positions are integrated (?).
    # (batch_size, zn_samples).

    # LOG SPACE
    free_energy = dists * density  # (batch_size, zn_samples).
    shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1, device=device), free_energy[:, :-1]],
                                    dim=-1)  # shift one step
    alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here. (Eq. 10)?
    transmittance = torch.exp(
        -torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now. (Eq. 4)
    weights = alpha * transmittance  # probability of the ray hits something here

    return weights


def volume_rendering_fg(z_vals: Tensor,
                        z_max: Tensor,
                        sdf_net: SDFNet,
                        sdf_val: Tensor,
                        device: torch.device) -> Tuple[Tensor, Tensor]:
    """ this function computes the volume of foreground and transmittance of background.
    Refer to: https://github.com/lioryariv/volsdf/blob/main/code/model/network_bg.py#L125.

    Args:
        z_vals (Tensor): sampled z according to the volsdf (b_size, zn_samples - 1).
        z_max (Tensor): the maximum value of z (b_size,).
        sdf_net (SDFNet): sdf-net.
        sdf_val (Tensor): sdf values of the sampled position (b_size * (zn_samples - 1), 1).
        device (torch.device): the device you are using.

    Returns:
        (Tensor): weights for each 3D position (batch_size, n_samples).
        (Tensor): transmittance for bg (batch_size,)
    """

    density_flat = sdf_net.density(sdf_val)
    density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size, zn_samples - 1)

    dists = z_vals[:, 1:] - z_vals[:, :-1]  # (batch_size, zn_samples - 2)
    dists = torch.cat([dists, z_max.unsqueeze(-1) - z_vals[:, -1:]], -1)  # (batch_size, zn_samples - 1)

    # LOG SPACE
    free_energy = dists * density  # (batch_size, zn_samples - 1)
    shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1, device=device), free_energy],
                                    dim=-1)  # add 0 for transparency 1 at t_0.
    alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
    transmittance = torch.exp(
        -torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
    fg_transmittance = transmittance[:, :-1]
    weights = alpha * fg_transmittance  # probability of the ray hits something here
    bg_transmittance = transmittance[:, -1]  # factor to be multiplied with the bg volume rendering

    return weights, bg_transmittance


def volume_rendering_bg(z_vals_bg: Tensor, sdf_val_bg: Tensor, device: torch.device) -> Tensor:
    """ this function computes the volume of background.
    Refer to: https://github.com/lioryariv/volsdf/blob/main/code/model/network_bg.py#L144.

    Args:
        z_vals_bg (Tensor): sampled z values (b_size, zn_samples_bg).
        sdf_val_bg (Tensor): sdf values (b_size * zn_samples_bg, 1).
        device (torch.device): the device you are using.

    Returns:
        (Tensor): calculated volume.

    Note:
        instead of using AbsDensity (https://github.com/lioryariv/volsdf/blob/main/code/model/density.py#L33),
        we simply use torch.abs() for computing the density background.
    """

    bg_density_flat = torch.abs(sdf_val_bg)  # (batch_size * zn_samples_bg, 1)
    bg_density = bg_density_flat.reshape(-1, z_vals_bg.shape[1])  # (batch_size, zn_samples_bg)

    bg_dists = z_vals_bg[:, :-1] - z_vals_bg[:, 1:]  # (batch_size, zn_samples_bg - 1)
    bg_dists = torch.cat([bg_dists, torch.tensor([1e10], device=device).unsqueeze(0).repeat(bg_dists.shape[0], 1)], -1)
    # (batch_size, zn_samples_bg)

    # LOG SPACE
    bg_free_energy = bg_dists * bg_density  # (batch_size, zn_samples_bg)
    bg_shifted_free_energy = torch.cat([torch.zeros(bg_dists.shape[0], 1, device=device), bg_free_energy[:, :-1]],
                                       dim=-1)  # shift one step
    bg_alpha = 1 - torch.exp(-bg_free_energy)  # probability of it is not empty here
    bg_transmittance = torch.exp(
        -torch.cumsum(bg_shifted_free_energy, dim=-1))  # probability of everything is empty up to now
    bg_weights = bg_alpha * bg_transmittance  # probability of the ray hits something here

    return bg_weights


def depth_to_pts_bg(rays_o: Tensor,
                    rays_d: Tensor,
                    depth: Tensor,
                    r: float) -> Tensor:
    """ this function computes x', y', z', and 1/r from 1/r.
    For the representation, refer to: https://arxiv.org/pdf/2010.07492.pdf, Fig. 8.
    For the source code, refer to: https://github.com/lioryariv/volsdf/blob/main/code/model/network_bg.py#L160.

    Args:
        rays_o (Tensor): the origin of the rays (b, n, 3).
        rays_d (Tensor): the directions of the rays (b, n, 3).
        depth (Tensor): sampled depths represented in [1/R, ..., 0] (b, n).
        r (float): radius of the bounding sphere.

    Returns:
        (Tensor): NeRF++ representation (b, n, 4).
    """

    b_, n_ = rays_o.shape[:2]

    rays_o_flat = rays_o.reshape(-1, 3)  # (b * n, 3)
    rays_d_flat = rays_d.reshape(-1, 3)  # (b * n, 3)

    o_dot_d = dot_product_tensor(rays_o_flat, rays_d_flat)  # (b * n, 1)
    d_sphere = get_sphere_intersections(rays_o=rays_o_flat, rays_d=rays_d_flat, r=r)  # (b * n, 1)

    p_sphere = rays_o_flat + d_sphere * rays_d_flat  # (b * n, 3)

    p_mid = rays_o_flat - o_dot_d * rays_d_flat  # (b * n, 3)
    p_mid_norm = torch.norm(p_mid, dim=-1).reshape(b_, n_)  # (b, n)

    rot_axis = normalize_tensor(torch.cross(rays_o_flat, p_sphere, dim=-1)).reshape(b_, n_, 3)  # (b, n, 3)
    phi = torch.asin(p_mid_norm / r)  # (b, n)
    theta = torch.asin(p_mid_norm * depth)

    rot_angle = (phi - theta).unsqueeze(-1)  # (b, n, 1)
    p_sphere = p_sphere.reshape(b_, n_, 3)  # (b, n, 3)

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(
        rot_angle) + rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    return pts


def fibonacci_sphere_sampling(normals: Tensor,
                              sample_num: int,
                              device: torch.device,
                              z_scale: float = 1.,
                              random_rotate: bool = True) -> Tensor:
    """ this function samples rays over the hemisphere according to Fibonacci sampling.
    Refer to: https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/model/ray_sampling.py

    Args:
        normals (Tensor): surface normal represented in the world coordinate (n, 3).
        sample_num (int): the number of rays you want to sample.
        device (torch.device): used device.
        z_scale (float): a scale number for limiting the sampling space. if one, it samples from the whole hemisphere.
            if larger than one, it narrows the zenith angles of the sampling point.
        random_rotate (bool): if true, randomly rotate azimuth angle of the samples.

    Returns:
        sampled rays (n, sample_num, 3).
    """

    if z_scale < 1.:
        raise ValueError("z_scale must be larger than or equal to 1.")

    n = normals.shape[0]

    delta = math.pi * (3. - math.sqrt(5.))

    # Fibonacci sphere sample around z axis.
    idx = torch.arange(sample_num, device=device).float().unsqueeze(0).repeat(n, 1)  # (n, sample_num)
    z = 1 - 2 * idx / (2 * sample_num - 1) / z_scale  # (n, sample_num)

    rad = torch.sqrt(1 - z ** 2)  # (n, sample_num)
    phi = delta * idx

    if random_rotate:
        phi = torch.rand(n, 1, device=device) * 2 * math.pi + phi

    y = torch.cos(phi) * rad  # (n, sample_num)
    x = torch.sin(phi) * rad  # (n, sample_num)

    z_samples = torch.stack([x, y, z], dim=-1).permute([0, 2, 1])  # (n, 3, sample_num)
    assert z_samples.shape == (n, 3, sample_num)

    # rotate to normal
    z_vector = torch.zeros_like(normals)  # (n, 3)
    z_vector[:, 2] = 1  # (n, 3)

    rotation_matrix = calc_rot_mat_from_two_vec(z_vector, normals)  # (n, 3, 3)

    incident_dirs = rotation_matrix @ z_samples  # (n, 3, sample_num)
    incident_dirs = incident_dirs.permute([0, 2, 1]).reshape(-1, 3)  # (n * sample_num, 3)
    incident_dirs = normalize_tensor(incident_dirs).reshape(n, sample_num, 3)  # (n, sample_num, 3)

    return incident_dirs


def sample_illum_from_env(env_map: Tensor, light_dir: Tensor, env_h: int, env_w: int) -> Tensor:
    """ sample the illumination from environment map.

    Args:
        env_map (torch.Tensor): environment map (h * w, 3).
        light_dir (torch.Tensor): lighting directions in global Euclid coordinate (b_size, sample_num, 3).
        env_h (int): height of the environment map.
        env_w (int): width of the environment map.

    Returns:
        (torch.Tensor) sampled illuminations (b_size, sample_num, 3).
    """

    b_size = light_dir.shape[0]
    s_num = light_dir.shape[1]

    x = light_dir[..., 0]  # (b_size, sample_num)
    y = light_dir[..., 1]  # (b_size, sample_num)
    z = light_dir[..., 2]  # (b_size, sample_num)

    theta = torch.acos(z / torch.sqrt(x ** 2 + y ** 2 + z ** 2))  # (b_size, sample_num), [0, pi]
    phi = torch.sign(y) * torch.acos(x / (torch.sqrt(x ** 2 + y ** 2) + 1e-07))  # (b_size, sample_num), [-pi, +pi]

    theta_idx = torch.round(theta / math.pi * (env_h - 1))  # (b_size, sample_num), [0, h-1]
    phi_idx = torch.round(phi / math.pi / 2. * (env_w - 1))  # (b_size, sample_num), [-(w-1)/2, (w-1)/2]

    idx = (theta_idx * env_w + phi_idx).to(torch.long)  # (b_size, sample_num)

    if torch.min(idx) < 0:
        idx[idx < 0] = 0

    env_map_expanded = env_map[None, :, :].expand(b_size, env_h * env_w, 3)
    idx_expanded = idx[:, :, None].expand(b_size, s_num, 3)

    extracted_env = torch.gather(env_map_expanded, dim=1, index=idx_expanded)

    return extracted_env
