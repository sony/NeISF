# shapeutils.py
""" Library for handling mesh (export ply, uv texture map, etc.).

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from typing import Tuple

import numpy as np
from numpy import ndarray

from skimage import measure
import torch
import trimesh

from mymodules.models.volsdf import SDFNet


class Groupby(object):
    """
    Refer to: https://github.com/Kai-46/IRON/blob/main/models/export_materials.py#L59.
    """

    def __init__(self, keys):
        """note keys are assumed to be integer"""
        super().__init__()

        self.unique_keys, self.keys_as_int = np.unique(keys, return_inverse=True)
        self.n_keys = len(self.unique_keys)
        self.indices = [[] for _ in range(self.n_keys)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]

    def apply(self, function, vector):
        assert len(vector.shape) <= 2
        if len(vector.shape) == 2:
            result = np.zeros((self.n_keys, vector.shape[-1]))
        else:
            result = np.zeros((self.n_keys,))

        for k, idx in enumerate(self.indices):
            result[k] = function(vector[idx], axis=0)

        return result


def get_grid_uniform(resolution: int, grid_boundary: list) -> dict:
    """ Generate uniform samples inside a boundary cube.

    refer to: https://github.com/lioryariv/volsdf/blob/main/code/utils/plots.py#L320

    Args:
        resolution (int): the resolution of sampling points.
        grid_boundary (list): the boundary of 3d samples. must be like [min, max].

    Returns:
        (dict): a dictionary that includes grid samples and other information.
    """

    if not isinstance(grid_boundary, list):
        raise TypeError("grid_boundary must be list.")

    if len(grid_boundary) != 2:
        raise ValueError("the length of grid_boundary must be 2.")

    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}


def get_surface_trace(sdf, resolution: int = 100, grid_boundary: list = None, level: int = 0):
    """ Extract a 3D mesh by using the marching cube algorithm.

    Args:
        sdf: SDF function.
        resolution (int): the resolution of sampling points for the marching cube algorithm.
        grid_boundary (list or None): the boundary of 3d samples. must be like [min, max].
        level (int): threshold for creating the mesh. in theory, this must be always zero.

    Returns:
        extracted 3D mesh.
    """

    if grid_boundary is None:
        grid_boundary = [-2.0, 2.0]

    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if np.min(z) > level or np.max(z) < level:  # all the sampled points locate inside or outside the object.
        raise ValueError("All the points locate outside or inside the object. Cannot extract mesh.")
    else:
        z = z.astype(np.float32)
        z = z.reshape((grid['xyz'][1].shape[0], grid['xyz'][0].shape[0], grid['xyz'][2].shape[0])).transpose([1, 0, 2])

        spacing = (grid['xyz'][0][2] - grid['xyz'][0][1],
                   grid['xyz'][0][2] - grid['xyz'][0][1],
                   grid['xyz'][0][2] - grid['xyz'][0][1])

        verts, faces, normals, values = measure.marching_cubes(volume=z, level=level, spacing=spacing)

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
        mesh = trimesh.Trimesh(verts, faces, normals)

    return mesh


def sample_surface(v: ndarray, f: ndarray, tc: ndarray, ftc: ndarray, n_samples: int = 5e06) -> Tuple[ndarray, ndarray]:
    """ Sample 3D points on the surface of the input polygon.

    the implementation refers to: https://github.com/Kai-46/IRON/blob/main/models/export_materials.py#L13.

    Args:
        v (ndarray): array of vertex positions of shape (v, 3).
        f (ndarray): f array of face indices into vertex positions of shape (f, 3).
        tc (ndarray): array of texture coordinates of shape (t, 2).
        ftc (ndarray): f array of face indices into vertex texture coordinates of shape (f, 3).
        n_samples (int): the number of sample point.
            For the default value, see: https://github.com/Kai-46/IRON/blob/main/models/export_materials.py#L176.

    Returns:
        ndarray: sampled 3D points of shape (s, 3).
        ndarray: sampled texture points of shape (s, 2).
    """

    # compute the areas of each face to determine the sample points.
    vec_cross = np.cross(v[f[:, 0], :] - v[f[:, 2], :], v[f[:, 1], :] - v[f[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
    face_areas = face_areas / np.sum(face_areas)

    # determine the number of sample points on each face.
    n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, int(floor_num), replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples_new = np.sum(n_samples_per_face)

    # create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples_new,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples_new, 2)

    aa = v[f[sample_face_idx, 0], :]
    bb = v[f[sample_face_idx, 1], :]
    cc = v[f[sample_face_idx, 2], :]
    pp = (1 - np.sqrt(r[:, 0:1])) * aa + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * bb + np.sqrt(r[:, 0:1]) * r[:, 1:] * cc

    aa = tc[ftc[sample_face_idx, 0], :]
    bb = tc[ftc[sample_face_idx, 1], :]
    cc = tc[ftc[sample_face_idx, 2], :]
    pp_uv = \
        (1 - np.sqrt(r[:, 0:1])) * aa + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * bb + np.sqrt(r[:, 0:1]) * r[:, 1:] * cc

    return pp.astype(np.float32), pp_uv.astype(np.float32)


def accumulate_splat_material(texture_image: ndarray, weight_image: ndarray, uv: ndarray, merged_texture: ndarray) \
        -> Tuple[ndarray, ndarray]:
    """ This function applies smoothing to the output uv texture by using Gaussian weight(?).
    Refer to: https://github.com/Kai-46/IRON/blob/main/models/export_materials.py#L84.

    Args:
        texture_image (ndarray): the uv texture of shape (h, w, n).
        weight_image (ndarray): the uv weight of shape (h, w).
        uv (ndarray): texture points (s, 2).
        merged_texture (ndarray): estimated texture values (s, n).

    Returns:
        ndarray: smoothed diffuse uv map.
    """

    # check the dimension of texture.
    if texture_image.shape[2] != merged_texture.shape[1]:
        raise ValueError("dimension of texture map and computed texture values are different.")

    uv_height, uv_width = texture_image.shape[:2]

    texture_image = texture_image.reshape((uv_height * uv_width, -1))  # (h * w, n)
    weight_image = weight_image.reshape((uv_height * uv_width,))  # (h * w)

    # label each 3d point with their splat pixel index
    uv[:, 0] = uv[:, 0] * uv_width
    uv[:, 1] = uv_height - uv[:, 1] * uv_height

    # repeat to a neighborhood
    merged_texture = np.tile(merged_texture, (5, 1))  # (5s, n)
    uv_up = np.copy(uv)
    uv_up[:, 1] -= 1
    uv_right = np.copy(uv)
    uv_right[:, 0] += 1
    uv_down = np.copy(uv)
    uv_down[:, 1] += 1
    uv_left = np.copy(uv)
    uv_left[:, 0] -= 1
    uv = np.concatenate((uv, uv_up, uv_right, uv_down, uv_left), axis=0)

    # compute pixel coordinates
    pixel_col = np.floor(uv[:, 0])
    pixel_row = np.floor(uv[:, 1])
    label = (pixel_row * uv_width + pixel_col).astype(int)

    # filter out-of-range points
    mask = np.logical_and(label >= 0, label < uv_height * uv_width)
    label = label[mask]
    uv = uv[mask]
    merged_texture = merged_texture[mask]
    pixel_col = pixel_col[mask]
    pixel_row = pixel_row[mask]

    # compute gaussian weight
    sigma = 1.0
    weight = np.exp(-((uv[:, 0] - pixel_col - 0.5) ** 2 + (uv[:, 1] - pixel_row - 0.5) ** 2) / (2 * sigma * sigma))

    groupby_obj = Groupby(label)
    delta_texture = groupby_obj.apply(np.sum, weight[:, np.newaxis] * merged_texture)
    delta_weight = groupby_obj.apply(np.sum, weight)

    texture_image[groupby_obj.unique_keys] += delta_texture
    weight_image[groupby_obj.unique_keys] += delta_weight

    texture_image = texture_image.reshape((uv_height, uv_width, -1))  # (h, w, n).
    weight_image = weight_image.reshape((uv_height, uv_width))

    return texture_image, weight_image


def ray_march(ro: torch.Tensor, rd: torch.Tensor, sdf: SDFNet, max_step: int = 100) -> torch.Tensor:
    """ This function calculates the intersection point between a ray (ro + t * rd) and a surface defined by SDF.

    Args:
        ro (torch.Tensor): the origin of the ray (n, 3).
        rd (torch.Tensor): the direction of the ray (n, 3).
        sdf (SDFNet): trained SDF.
        max_step (int): maximum iteration.

    Returns:
        distance from the ray origin (ro) to the surface with shape (n, 1).
    """

    t = torch.zeros(ro.shape[0], 1, device=ro.device)

    for step in range(max_step):
        p = ro + t * rd
        t += sdf.get_sdf_vals(x=p, bounding_sphere_r=3.0)

    return t
