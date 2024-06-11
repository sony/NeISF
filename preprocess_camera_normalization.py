# preprocess_camera_normalization.py
""" This script normalizes the cameras in your dataset.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

USAGE:
    $ python preprocess_camera_normalization.py --flist {YOUR DIR1} {YOUR DIR2} {YOUR_ DIR3} ...

NOTE:
    - This script will compute the target point by using the z-axes of all the cameras, shift the point to the origin,
    and then normalize all the cameras.
    - If you want to normalize several directories at the same time, for instance you have a training scene and
    an evaluation scene, please input multiple directories as in the example above.
    - The original `poses_bounds.npy` will be saved in the same directory with the name of
    `poses_bounds_before_normalization.npy`.
"""

import argparse
import os
from pathlib import Path

import numpy as np

from mymodules import renderutils
from mymodules.globals import BOUNDING_SPHERE_R


BASE_PATH = Path("images")

parser = argparse.ArgumentParser()
parser.add_argument("--flist", required=True, nargs="*", type=str, help="folder names.")


if __name__ == '__main__':
    args = parser.parse_args()
    folder_name_list = args.flist

    # load all the `poses_bounds.npy`.
    poses_bounds_all = np.empty((0, 17))
    scene_num_list = []

    for ii, folder_name in enumerate(folder_name_list):
        scene_path = BASE_PATH.joinpath(folder_name)

        if scene_path.joinpath("poses_bounds_before_normalization.npy").exists():
            raise ValueError("your directory seems to be already normalized.")

        poses_bounds_scene = np.load(str(scene_path.joinpath("poses_bounds.npy")))
        scene_num_list.append(poses_bounds_scene.shape[0])
        poses_bounds_all = np.append(poses_bounds_all, poses_bounds_scene, axis=0)

    # change the shape of poses_bounds
    poses_all = poses_bounds_all[:, :15].reshape(-1, 3, 5)  # (n, 3, 5)
    height, width, focal = renderutils.get_hwf_from_poses_bounds(poses_all)

    # change the coordinate
    poses_all = np.concatenate([poses_all[..., 1:2], -poses_all[..., :1], poses_all[..., 2:4]], -1)  # (n, 3, 4)

    # calculate the canter and normalization matrix
    center = renderutils.get_center_point(poses_all)
    normalization_mat = renderutils.get_average_poses(poses_all)

    normalization_mat_homo = np.eye(4)
    normalization_mat_homo[:3, :] = normalization_mat
    normalization_mat_homo[:3, 3] = center

    # rotate and shift the poses.
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses_all), 1, 1))  # (n, 1, 4)
    poses_all_homo = np.concatenate([poses_all, last_row], 1)  # (n, 4, 4)

    poses_all_centered = np.linalg.inv(normalization_mat_homo) @ poses_all_homo  # (n, 4, 4)
    poses_all_centered = poses_all_centered[:, :3, :]  # (n, 3, 4)

    # normalize the scale to locate all the cameras within a sphere of radius 3.
    # refer to: https://arxiv.org/abs/2106.12052) B.1.
    cam_loc_max = np.max(np.linalg.norm(poses_all_centered[:, :, 3], axis=1))
    poses_all_centered[..., 3] *= BOUNDING_SPHERE_R / cam_loc_max / 1.1

    # update poses_bounds
    for scene_num, folder_name in zip(scene_num_list, folder_name_list):
        scene_path = BASE_PATH.joinpath(folder_name)
        poses_bounds_updated = []

        jj = 0
        while jj < scene_num:
            poses_all_centered_sub = poses_all_centered[0, :, :]  # (3, 4)

            c2w = poses_all_centered_sub[:, :3]  # (3, 3)
            cam_pos = poses_all_centered_sub[:, 3]  # (3,)

            cam_vec_x = -c2w[:, 1].flatten()  # (3,)
            cam_vec_y = c2w[:, 0].flatten()  # (3,)
            cam_vec_z = c2w[:, 2].flatten()  # (3,)

            poses_bounds_updated_sub = np.zeros((17,), dtype=np.float32)
            poses_bounds_updated_sub[0] = cam_vec_x[0]
            poses_bounds_updated_sub[1] = cam_vec_y[0]
            poses_bounds_updated_sub[2] = cam_vec_z[0]
            poses_bounds_updated_sub[3] = cam_pos[0]
            poses_bounds_updated_sub[5] = cam_vec_x[1]
            poses_bounds_updated_sub[6] = cam_vec_y[1]
            poses_bounds_updated_sub[7] = cam_vec_z[1]
            poses_bounds_updated_sub[8] = cam_pos[1]
            poses_bounds_updated_sub[10] = cam_vec_x[2]
            poses_bounds_updated_sub[11] = cam_vec_y[2]
            poses_bounds_updated_sub[12] = cam_vec_z[2]
            poses_bounds_updated_sub[13] = cam_pos[2]
            poses_bounds_updated_sub[4] = height
            poses_bounds_updated_sub[9] = width
            poses_bounds_updated_sub[14] = focal

            poses_bounds_updated.append(poses_bounds_updated_sub)

            # delete the first row.
            poses_all_centered = np.delete(poses_all_centered, 0, axis=0)
            jj += 1

        # change the name of the target scene's `poses_bounds.npy` to `poses_bounds_before_normalization.npy`
        os.rename(str(scene_path.joinpath("poses_bounds.npy")),
                  str(scene_path.joinpath("poses_bounds_before_normalization.npy")))

        # save the updated `poses_bounds.npy`
        assert not scene_path.joinpath("poses_bounds.npy").exists()
        np.save(str(scene_path.joinpath("poses_bounds.npy")), np.array(poses_bounds_updated))
