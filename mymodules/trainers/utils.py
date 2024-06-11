# trainers/utils.py
""" Library for utility functions of the training.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import math
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.activation import Sigmoid, Softplus

EPS = 1e-07


def assign_device(gpu_num: int) -> torch.device:
    """ Returns devices according to the number of GPUs.

    Args:
          gpu_num (int): the number of gpus you want to use. if you do not have any GPU, input 0.

    Returns:
          torch.device: assigned device.
    """

    if not isinstance(gpu_num, int):
        raise TypeError("gpu_num must be int.")
    if gpu_num < 0:
        raise ValueError("gpu_num must be 0 or positive integer.")

    if gpu_num == 0:  # no gpu
        device = torch.device("cpu")
    elif gpu_num == 1:  # one gpu.
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:  # multiple gpus.
        raise NotImplementedError("currently this code cannot handle multi-GPUs.")

    return device


def activation_func_provider(dataset_type: str) -> Sigmoid or Softplus:
    """ Return an activation function according to the dataset type.

    Args:
        dataset_type (str): dataset type, only `neisf` is allowed.

    Returns:
        activation function.
    """

    if dataset_type == "neisf":
        return Softplus()
    else:
        raise ValueError(f"wrong data-type. only `neisf` is accepted. yours: {dataset_type}.")


def generate_env_map_coordinate(u_res: int, v_res: int) -> Tensor:
    """ Return ray directions for rendering environment map.

    Args:
        u_res (int): vertical resolution of the env map.
        v_res (int): horizontal.

    Returns:
        computed ray directions.
    """

    v_mat, u_mat = torch.meshgrid(
        torch.linspace(0, math.pi, v_res), torch.linspace(0, 2 * math.pi, u_res), indexing="ij")
    dx = torch.reshape(torch.cos(u_mat) * torch.sin(v_mat), (v_res * u_res, 1))
    dy = torch.reshape(torch.sin(u_mat) * torch.sin(v_mat), (v_res * u_res, 1))
    dz = torch.reshape(torch.cos(v_mat), (v_res * u_res, 1))

    rays_d_all = torch.cat((dx, dy, dz), dim=1)  # (u_res * v_res, 3)

    return rays_d_all


def replace_nans_infs_with_tiny(model: nn.Module):
    """ Replace `nan` and `ìnf` to tiny values.
    """
    with torch.no_grad():
        for param in model.parameters():
            mask_nan = torch.isnan(param)
            mask_inf = torch.isinf(param)
            mask = mask_nan | mask_inf
            param[mask] = EPS


def trainer_provider(config_path: Path,
                     is_train: bool,
                     inference_folder: str = None,
                     inference_batch: int = None,
                     inference_illum_sample: int = None):
    """ Returns an initialized trainer.

    Args:
        config_path (Path): a file path to the config file.
        is_train (bool): True for training. False otherwise.
        inference_folder (str): the name of the folder you want to inference.
        inference_batch (int): the batch size for inference.
        inference_illum_sample (int): the number of sampled illumination for inference.

    Returns:
        the initialized trainer.

    Note:
        We import each trainer class inside this function to avoid circular import.
    """

    # load config file.
    with open(config_path, "r") as f:
        configs = json.load(f)

    # change some variables according to train/inference.
    data_dir = configs["data_dir"] if is_train or inference_folder is None else inference_folder
    batch_size = configs["batch_size"] if is_train or inference_batch is None else inference_batch

    # initialize the trainer.
    if configs["trainer_name"] == "TrainerVolSDF":
        from .trainers_first_stage import TrainerVolSDF
        trainer = TrainerVolSDF(
            is_training=is_train,
            data_dir=data_dir,
            dataset_type=configs["dataset_type"],
            experiment_name=configs["experiment_name"],
            batch_size=batch_size,
            max_epoch=configs["max_epoch"],
            sample_num=configs["sample_num"],
            positional_encoding_x_res=configs["positional_encoding_x_res"],
            positional_encoding_d_res=configs["positional_encoding_d_res"],
            gpu_num=configs["gpu_num"],
            lr=configs["lr"],
            weight_dic=configs["weights"],
            config_path=config_path,
            use_mask=configs["use_mask"],
        )

    elif configs["trainer_name"] == "TrainerNeISF":
        illum_sample_num = \
            configs["illum_sample_num"] if is_train or inference_illum_sample is None else inference_illum_sample

        from . import TrainerNeISF
        trainer = TrainerNeISF(
            is_training=is_train,
            data_dir=data_dir,
            dataset_type=configs["dataset_type"],
            experiment_name=configs["experiment_name"],
            batch_size=configs["batch_size"],
            max_epoch=configs["max_epoch"],
            sample_num=configs["sample_num"],
            illum_sample_num=illum_sample_num,
            positional_encoding_x_res=configs["positional_encoding_x_res"],
            positional_encoding_d_res=configs["positional_encoding_d_res"],
            gpu_num=configs["gpu_num"],
            lr=configs["lr"],
            weight_dic=configs["weights"],
            stage_name=configs["stage_name"],
            config_path=config_path,
            previous_stage_dir=configs["previous_stage_dir"],
            previous_stage_epoch_num=configs["previous_stage_epoch_num"],
            use_mask=configs["use_mask"],
            max_step_ray_march=configs["max_step_ray_march"],
        )

    elif configs["trainer_name"] == "TrainerNeISFNoStokes":
        illum_sample_num = \
            configs["illum_sample_num"] if is_train or inference_illum_sample is None else inference_illum_sample

        from . import TrainerNeISFNoStokes
        trainer = TrainerNeISFNoStokes(
            is_training=is_train,
            data_dir=data_dir,
            dataset_type=configs["dataset_type"],
            experiment_name=configs["experiment_name"],
            batch_size=configs["batch_size"],
            max_epoch=configs["max_epoch"],
            sample_num=configs["sample_num"],
            illum_sample_num=illum_sample_num,
            positional_encoding_x_res=configs["positional_encoding_x_res"],
            positional_encoding_d_res=configs["positional_encoding_d_res"],
            gpu_num=configs["gpu_num"],
            lr=configs["lr"],
            weight_dic=configs["weights"],
            stage_name=configs["stage_name"],
            config_path=config_path,
            previous_stage_dir=configs["previous_stage_dir"],
            previous_stage_epoch_num=configs["previous_stage_epoch_num"],
            use_mask=configs["use_mask"],
            max_step_ray_march=configs["max_step_ray_march"],
        )

    else:
        raise ValueError("Wrong trainer name!!")
    return trainer
