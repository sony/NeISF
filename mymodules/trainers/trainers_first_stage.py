# trainers_first_stage.py
""" This script defines some trainer classes for the 1st-stage trainings.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .trainers_base import TrainerBase
from .utils import activation_func_provider

import mymodules.models.volsdf as volsdf

from mymodules import embedders
from mymodules import polarutils
from mymodules.datasets import DataPrefetcher
from mymodules.imageutils import write_all_images_in_dict, MAX_16BIT
from mymodules.renderutils import volume_rendering
from mymodules.tensorutils import calc_weighted_sum, normalize_tensor


class TrainerVolSDF(TrainerBase):
    """ Trainer for reproducing VolSDF.

    Attributes:
        embedder_dir (Embedder): positional encoding for ray direction.
        loss_rgb (torch.nn.modules.loss): a loss function for rgb value.
    """

    def __init__(self,
                 is_training: bool,
                 data_dir: str,
                 dataset_type: str,
                 experiment_name: str,
                 batch_size: int,
                 max_epoch: int,
                 sample_num: int,
                 positional_encoding_x_res: int,
                 positional_encoding_d_res: int,
                 gpu_num: int,
                 lr: float,
                 weight_dic: dict,
                 config_path: Path,
                 use_mask: bool):

        super(TrainerVolSDF, self).__init__(
            is_training=is_training,
            data_dir=data_dir,
            dataset_type=dataset_type,
            experiment_name=experiment_name,
            batch_size=batch_size,
            max_epoch=max_epoch,
            sample_num=sample_num,
            positional_encoding_x_res=positional_encoding_x_res,
            gpu_num=gpu_num,
            lr=lr,
            weight_dic=weight_dic,
            config_path=config_path,
            use_mask=use_mask,
        )

        self.embedder_dir = embedders.PEEmbedder(l_num=positional_encoding_d_res)

        # set up the rendering net.
        last_activation_func = activation_func_provider(dataset_type=dataset_type)
        self.models_dic["rendernet"] = volsdf.RenderNet(
            depth=4,
            width=256,
            in_ch=6 + self.embedder_dir.out_dim + self.shape_feature_vector_size,
            out_ch=3,
            embedder=self.embedder_dir,
            last_activation=last_activation_func,
        )

        self.loss_rgb = nn.L1Loss()

    def train_one_epoch(self, data_loader: DataLoader) -> float:
        loss_sum = 0

        prefetcher = DataPrefetcher(data_loader, self.device)

        for current_batch_num, batch in enumerate(prefetcher):
            s0 = batch["s0"]  # (b_size, 3)
            rays_o = batch["rays_o"]  # (b_size, 3)
            rays_d = batch["rays_d"]  # (b_size, 3)

            rgb_values, z_samples_eik, _ = self.forward(rays_o, rays_d, self.is_training)

            # calc rgb loss.
            loss_s0 = self.loss_rgb(rgb_values, s0)

            # calc Eikonal loss.
            grad_theta = self.calc_grad_theta(rays_o, rays_d, z_samples_eik)
            loss_ek = self.loss_eikonal(grad_theta)

            loss = loss_s0 + self.eik_weight * loss_ek
            loss_sum += loss.item()

            self.update_all_models(loss, clip_sdf_grad=False)

            print(
                f"epoch: {self.current_epoch:03d}, "
                f"batch: {current_batch_num + 1:06d}/{self.batch_num:06d}, "
                f"loss: {loss.item():.4f} \r"
            )

        return loss_sum

    def inference_sub(self, save_path: Path, split_size: int):
        for img_idx in range(self.data.n_images):
            batch_dic = self.data.getitem_by_frame_id_split(img_idx, split_size)
            rays_o_split = batch_dic["rays_o"]
            rays_d_split = batch_dic["rays_d"]
            w2c_split = batch_dic["w2c"]

            w2c = w2c_split[0][0].to(self.device)

            result_rgb = []
            result_normal = []

            for rays_o, rays_d in zip(rays_o_split, rays_d_split):
                rays_o = rays_o.to(self.device)
                rays_d = rays_d.to(self.device)

                rgb_values, z_samples_eik, normals = self.forward(rays_o, rays_d, self.is_training)

                # convert normal
                normals = polarutils.rotate_normal_from_world_to_tangent(w2c=w2c, rays_d=rays_d, normals=normals)
                normals = (normals + 1.) / 2.

                result_rgb.append(rgb_values.detach().cpu())
                result_normal.append(normals.detach().cpu())

            result_rgb = torch.cat(result_rgb, dim=0)
            result_normal = torch.cat(result_normal, dim=0)

            if self.use_mask:
                mask = self.data.mask_pool[img_idx]
            else:
                mask = None

            images_dic = self.reshape_and_scale_all_images(
                valid_mask=mask, max_val=MAX_16BIT, rgb=result_rgb, normal=result_normal,
            )
            write_all_images_in_dict(save_path, img_idx, images_dic)

    def forward(self, rays_o: Tensor, rays_d: Tensor, is_training: bool) -> Tuple[Tensor, Tensor, Tensor]:
        # Sampling the positions along the rays.
        z_vals, z_samples_eik = self.sampler.get_z_vals(rays_o, rays_d, self.models_dic["sdfnet"], is_training)
        zn_samples = z_vals.shape[1]

        rays_d_usq = rays_d.unsqueeze(1)  # (b_size, 1, 3)
        positions = (rays_o.unsqueeze(1) + rays_d_usq * z_vals.unsqueeze(2)).reshape(-1, 3)  # (b_size * zn_samples, 3)

        # Input the positions to SDFNet.
        sdf_val, feature_vec, gradients = \
            self.models_dic["sdfnet"].get_all_outputs(positions, self.data.bounding_sphere_r, is_training)

        rays_d_ = rays_d_usq.repeat(1, zn_samples, 1)
        rays_d_flat = rays_d_.reshape(-1, 3)  # (b_size * zn_samples, 3)

        rgb_flat = self.models_dic["rendernet"](positions, gradients, rays_d_flat, feature_vec)
        rgb = rgb_flat.reshape(-1, zn_samples, 3)  # (b_size, zn_samples, 3)

        weights = volume_rendering(z_vals, self.models_dic["sdfnet"], sdf_val, self.device)  # (b_size, zn_samples)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)  # (b_size, 3)

        if is_training:
            normals = None
        else:  # compute normal only when testing.
            normal_flat = normalize_tensor(gradients)  # (b_size * zn_samples, 3)
            normals = calc_weighted_sum(weights=weights, target=normal_flat)

        return rgb_values, z_samples_eik, normals
