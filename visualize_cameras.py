# visualize_cameras.py
""" This script visualizes all the cameras in your dataset.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

USAGE:
    `$ python visualize_cameras.py {DIR_NAME} {DATA_TYPE}`

Notes:
    currently, only `neisf` is allowed as DATA_TYPE.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mymodules.datasets import NeISFDataset
from mymodules.imageutils import my_write_plt_fig


parser = argparse.ArgumentParser()
parser.add_argument("dir_name", type=str, help="name of the directory you want to see the cameras.")
parser.add_argument("data_type", choices=["neisf"])


if __name__ == '__main__':
    args = parser.parse_args()
    dir_name = args.dir_name
    data_type = args.data_type

    scene_path = Path("images").joinpath(dir_name)

    if data_type == "neisf":
        dataset = NeISFDataset(scene_path.name, use_mask=False)
    else:
        raise ValueError(f"wrong dataset type: {data_type}")

    x_vec = torch.zeros((dataset.n_images, 3))
    y_vec = torch.zeros((dataset.n_images, 3))
    z_vec = torch.zeros((dataset.n_images, 3))
    t_vec = torch.zeros((dataset.n_images, 3))

    for i in range(dataset.n_images):
        batch_dic = dataset.getitem_by_frame_id(i)

        rays_o = batch_dic["rays_o"]  # (h * w, 3)
        w2c = batch_dic["w2c"]  # (h * w, 3, 3)

        c2w = w2c[0].T  # (3, 3)

        x_vec[i, :] = c2w[:, 0].flatten()
        y_vec[i, :] = c2w[:, 1].flatten()
        z_vec[i, :] = c2w[:, 2].flatten()
        t_vec[i, :] = rays_o[0, :]

    # plot cameras
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)
    ax.set_zlabel("z", size=14)

    scale = 3.5
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)
    ax.set_box_aspect((2 * scale, 2 * scale, 2 * scale))

    vec_scale = 0.3
    ax.quiver(
        t_vec[:, 0], t_vec[:, 1], t_vec[:, 2], z_vec[:, 0], z_vec[:, 1], z_vec[:, 2],
        length=vec_scale, normalize=True, color="b"
    )
    ax.quiver(
        t_vec[:, 0], t_vec[:, 1], t_vec[:, 2], y_vec[:, 0], y_vec[:, 1], y_vec[:, 2],
        length=vec_scale, normalize=True, color="g"
    )
    ax.quiver(
        t_vec[:, 0], t_vec[:, 1], t_vec[:, 2], x_vec[:, 0], x_vec[:, 1], x_vec[:, 2],
        length=vec_scale, normalize=True, color="r"
    )

    my_write_plt_fig(fig, scene_path.joinpath("camera_position.png"))
    plt.show()
    plt.close(fig)
