# datasets.py
""" This script defines some dataset classes.
Some lines may refer to: https://github.com/kwea123/nerf_pl.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mymodules import renderutils

from mymodules.globals import BOUNDING_SPHERE_R
from mymodules.imageutils import my_read_image, MAX_16BIT


class DataPrefetcher:
    """ Prefetch dataloader.

    Some parts may refer to:
        https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py#L265.

    Attributes:
        loader: an iterator created from a dataloader.
        batch (None or dict): a batch already located on a CUDA device.
        stream (torch.cuda.Stream): https://pytorch.org/docs/stable/generated/torch.cuda.stream.html
    """

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = iter(loader)
        self.batch = None
        self.device = device
        self.stream = torch.cuda.Stream()

        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != "meta":
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        torch.cuda.current_stream().wait_stream(self.stream)

        if self.batch is None:
            raise StopIteration
        else:
            batch = self.batch
            self.preload()

        return batch


class MyDataset(Dataset):
    """ My dataset class.

    We assume that the camera positions are already normalized to be inside a sphere of radius three,
    centered at the origin, like in VolSDF (https://arxiv.org/abs/2106.12052, p.15, B.1).
    And note that the near and far bound are always zero and three, respectively.

    Attributes:
        data_path (Path): path to the dataset from the working directory.
        bounding_sphere_r (float): a scene is assumed to be located within a sphere of this radius.
        near (float): a near bound of the scene. we always use zero.
        far (float): a far bound of the scene. we always use 2 * bounding_sphere_r.
        use_mask (bool): if you consider masks, use True.
        height (int): height of target scene.
        width (int): width of target scene.
        focal (float): focal length of target scene.
        s0_pool (:obj:`list` of :obj:`torch.Tensor`): a list of s0 images.
        s1_pool (:obj:`list` of :obj:`torch.Tensor`): a list of s1 images.
        s2_pool (:obj:`list` of :obj:`torch.Tensor`): a list of s2 images.
        w2c_pool (:obj:`list` of :obj:`torch.Tensor`): a list of world-to-camera matrices.
        mask_pool (:obj:`list` of :obj:`torch.Tensor`): a list of world-to-camera matrices.
        rays_o_pool (:obj:`list` of :obj:`torch.Tensor`): a list of origins of rays.
        rays_d_pool (:obj:`list` of :obj:`torch.Tensor`): a list of directions of rays.
        camera_axis_pool (:obj:`list` of :obj:`torch.Tensor`): a list of camera's x-axes (pointing right).
        n_images (int): the number of images.

    Note:
        your dataset class must be implemented with `load_data` function.
    """

    def __init__(self, data_dir: str, use_mask: bool):
        """
        Args:
            data_dir (str): the name of the directory.
            use_mask (bool): if you consider masks, use True.
        """
        self.data_path = Path("images").joinpath(data_dir)

        self.bounding_sphere_r = BOUNDING_SPHERE_R
        self.near = 0.
        self.far = self.bounding_sphere_r * 2.

        self.use_mask = use_mask

        self.height = None
        self.width = None
        self.focal = None
        self.s0_pool = []
        self.s1_pool = []
        self.s2_pool = []
        self.mask_pool = []
        self.w2c_pool = []
        self.rays_o_pool = []
        self.rays_d_pool = []
        self.camera_axis_pool = []

        self.load_data()

        # check initialization.
        if (
            self.height is None or
            self.width is None or
            self.focal is None or
            len(self.s0_pool) == 0 or
            len(self.s1_pool) == 0 or
            len(self.s2_pool) == 0 or
            len(self.mask_pool) == 0 or
            len(self.w2c_pool) == 0 or
            len(self.rays_o_pool) == 0 or
            len(self.rays_d_pool) == 0
        ):
            raise ValueError("Dataset initialization failed. Check `load_data()`.")

        self.n_images = len(self.s0_pool)

        # change the shape from the list of tensors to the tensors of shape (n*h*w, 3).
        self.s0_train = torch.cat(self.s0_pool, dim=0)
        self.s1_train = torch.cat(self.s1_pool, dim=0)
        self.s2_train = torch.cat(self.s2_pool, dim=0)
        self.mask_train = torch.cat(self.mask_pool, dim=0)
        self.w2c_train = torch.cat(self.w2c_pool, dim=0)
        self.camera_axis_train = torch.cat(self.camera_axis_pool, dim=0)
        self.rays_o_train = torch.cat(self.rays_o_pool, dim=0)
        self.rays_d_train = torch.cat(self.rays_d_pool, dim=0)

        if use_mask:  # sample only valid pixels.
            mask = self.mask_train[:, 0]  # (N,)

            self.s0_train = self.s0_train[mask]  # (N, 3)
            self.s1_train = self.s1_train[mask]  # (N, 3)
            self.s2_train = self.s2_train[mask]  # (N, 3)
            self.w2c_train = self.w2c_train[mask]  # (N, 3, 3)
            self.camera_axis_train = self.camera_axis_train[mask]  # (N, 3)
            self.rays_o_train = self.rays_o_train[mask]  # (N, 3)
            self.rays_d_train = self.rays_d_train[mask]  # (N, 3)

    def load_data(self):
        """ this function must set values for height, width, and focal,
        and append s0, s1, s2, w2c, rays_o, rays_d, and pixel_idx for each list.
        """
        raise NotImplementedError

    def getitem_by_frame_id(self, idx: int) -> dict:
        batch_dic = {
            "s0": self.s0_pool[idx],
            "s1": self.s1_pool[idx],
            "s2": self.s2_pool[idx],
            "w2c": self.w2c_pool[idx],
            "rays_o": self.rays_o_pool[idx],
            "rays_d": self.rays_d_pool[idx],
            "camera_axis": self.camera_axis_pool[idx],
            "mask": self.mask_pool[idx]
        }
        return batch_dic

    def getitem_by_frame_id_split(self, idx: int, split_size: int) -> dict:
        batch_dic = {
            "s0": torch.split(self.s0_pool[idx], split_size),
            "s1": torch.split(self.s1_pool[idx], split_size),
            "s2": torch.split(self.s2_pool[idx], split_size),
            "w2c": torch.split(self.w2c_pool[idx], split_size),
            "rays_o": torch.split(self.rays_o_pool[idx], split_size),
            "rays_d": torch.split(self.rays_d_pool[idx], split_size),
            "camera_axis": torch.split(self.camera_axis_pool[idx], split_size),
            "mask": torch.split(self.mask_pool[idx], split_size)
        }
        return batch_dic

    def __len__(self):
        """ Returns the length of the dataset.
        The returned value equivalents to the number of available rays.
        """
        return len(self.s0_train[:, 0])

    def __getitem__(self, idx):
        batch_dic = {
            "s0": self.s0_train[idx],  # (b_size, 3)
            "s1": self.s1_train[idx],  # (b_size, 3)
            "s2": self.s2_train[idx],  # (b_size, 3)
            "w2c": self.w2c_train[idx],  # (b_size, 3, 3)
            "camera_axis": self.camera_axis_train[idx],
            "rays_o": self.rays_o_train[idx],  # (b_size, 3)
            "rays_d": self.rays_d_train[idx],  # (b_size, 3)
        }
        return batch_dic


class NeISFDataset(MyDataset):
    """ NeISF dataset class.

    the folder structure should be as follows:
        images/
            |- data_dir
                |- images_s0/  img_001.exr, img_002.exr, ...
                |- images_s1/
                |- images_s2/
                |- masks/
                |- poses_bounds.npy

    Notes:
        for more details, please see README.
    """

    def __init__(self, data_dir: str, use_mask: bool):
        super(NeISFDataset, self).__init__(data_dir=data_dir, use_mask=use_mask)

    def load_data(self):
        poses_bounds = np.load(str(self.data_path.joinpath("poses_bounds.npy")))  # (n_images, 17)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (n_images, 3, 5).

        height, width, focal = renderutils.get_hwf_from_poses_bounds(poses)
        self.height = height
        self.width = width
        self.focal = focal

        # generate rays in a camera coordinate.
        rays_cam_coord = renderutils.generate_rays_cam_coord(height=height, width=width, focal=focal)  # (height, width, 3)

        # The original poses are "down right back", so change to " right up back".
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)  # (n_images, 3, 4)

        img_num = len(list(self.data_path.joinpath("images_s0").glob("*.exr")))

        # scene checks.
        if np.max(np.linalg.norm(poses[:, :, 3], axis=1)) > self.bounding_sphere_r * 1.1:
            raise ValueError("your camera is not normalized. please run `preprocess_camera_normalization.py`.")
        if img_num != poses.shape[0]:
            raise ValueError(
                f"the number of images and poses must be the same: n_image is {img_num}, n_pose is {poses.shape[0]}"
            )

        for idx in range(img_num):
            # store images_s0, images_s1, and images_s2 files.
            s0 = my_read_image(self.data_path.joinpath("images_s0", "img_{:03d}.exr".format(idx + 1)))
            s1 = my_read_image(self.data_path.joinpath("images_s1", "img_{:03d}.exr".format(idx + 1)))
            s2 = my_read_image(self.data_path.joinpath("images_s2", "img_{:03d}.exr".format(idx + 1)))
            mask_img = my_read_image(self.data_path.joinpath("masks", "img_{:03d}.png".format(idx + 1))) / MAX_16BIT

            s0 = torch.from_numpy(s0).reshape(-1, 3)
            s1 = torch.from_numpy(s1).reshape(-1, 3)
            s2 = torch.from_numpy(s2).reshape(-1, 3)
            mask = torch.from_numpy(mask_img > 0.999).reshape(-1, 3)

            self.s0_pool.append(s0)  # (h * w, 3)
            self.s1_pool.append(s1)
            self.s2_pool.append(s2)
            self.mask_pool.append(mask)

            # store world to camera matrices.
            c2w = torch.FloatTensor(poses[idx])  # (3, 4)
            w2c = c2w[:, :3].T  # (3, 3)
            self.w2c_pool.append(w2c.expand(height * width, 3, 3))

            # store the forward camera axis in world coordinate
            camera_axis = torch.tensor([[0., 1., 0.]]) @ w2c  # right axis of the camera in the world coordinate (1, 3)
            self.camera_axis_pool.append(camera_axis.expand(height * width, 3))

            # store rays origin and direction.
            rays_o, rays_d = renderutils.rotate_rays_to_world_coord(rays_cam_coord, c2w)
            self.rays_o_pool.append(rays_o)  # (h * w, 3)
            self.rays_d_pool.append(rays_d)  # (h * w, 3)


def get_dataset(dataset_type: str, data_dir: str, use_mask: bool) -> MyDataset:
    """ Return initialized dataset.

    Args:
        dataset_type: must be one of the followings: "neisf".
        data_dir: directory name.
        use_mask: True when using mask.

    Returns:
        initialized dataset class.
    """

    if dataset_type == "neisf":
        data = NeISFDataset(data_dir=data_dir, use_mask=use_mask)
    else:
        raise ValueError(f"wrong data-type. only neisf is accepted. yours: {dataset_type}")

    return data

