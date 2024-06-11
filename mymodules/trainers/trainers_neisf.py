# trainers_neisf.py
""" This script defines NeISF and NeISF-no-pol trainers.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from pathlib import Path

import igl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainers_base import TrainerBase, BlenderExportMixin

import mymodules.models.neisf as neisf

from mymodules import brdfutils
from mymodules import embedders
from mymodules import polarutils
from mymodules import shapeutils
from mymodules import renderutils

from mymodules.datasets import DataPrefetcher, get_dataset
from mymodules.globals import ENV_MAP_FOLDER_NAME
from mymodules.imageutils import my_read_image, my_write_image_exr, my_write_image_16bit, my_write_image_8bit, MAX_16BIT, MAX_8BIT
from mymodules.tensorutils import calc_weighted_sum, dot_product_tensor, normalize_tensor, gamma
from mymodules.shapeutils import ray_march


def _write_all_images_in_dict_for_neisf(save_path: Path, idx: int, images_dict: dict) -> bool:
    """ Save all the images in the input dict.
    Unlike write_all_images_in_dict in imageutils.py, this function also saves ".exr" format for s0, s1, and s2.

    Args:
      save_path (Path): The folder path you want to save all the images.
      idx (int): It will be included in your image name. It must be zero or positive integer, smaller than 999.
      images_dict (dict): Dict whose keys are image names and values are the images.

    Returns:
        bool: True if all the images are correctly saved, otherwise false.
    """

    if not isinstance(idx, int):
        raise TypeError("idx must be int.")
    if idx < 0:
        raise ValueError("idx must be larger than zero.")
    if idx + 1 > 999:
        raise ValueError("idx must be smaller than 999.")

    all_image_is_saved = True
    for image_name in images_dict:
        img = images_dict[image_name].detach().numpy()

        image_is_saved = my_write_image_16bit(
            save_path.joinpath(f"{idx + 1:03d}_{image_name}.png"),
            np.clip(img * MAX_16BIT, 0., MAX_16BIT),
        )
        all_image_is_saved *= image_is_saved

        if image_name in ["s0", "s1", "s2"]:  # also save exr file.
            image_is_saved = my_write_image_exr(
                save_path.joinpath(f"{idx + 1:03d}_{image_name}.exr"),
                images_dict[image_name].detach().numpy(),
            )
            all_image_is_saved *= image_is_saved

    return all_image_is_saved


class TrainerNeISF(BlenderExportMixin, TrainerBase):
    """ Trainer for Neural Incident Stokes Field (NeISF).

    This trainer has six MLPs for SDF, incident_s0, incident_diff, incident_spec, albedo, and roughness.
    This trainer computes the specular and diffuse Stokes vector using a physically-based polarization model.

    Attributes:
        stage_name ("init" or "joint"): a flag to define the training stage. must be `init` or `joint`.
        illum_sample_num (int): the number of samples for the incident light.
        max_step_ray_march (int): the number of steps for the ray marching.
        stokes_weight (float): a weight for compensating S1 and S2.
        loss_stokes (torch.nn.modules.loss): a loss function for stokes values.

    Notes:
        We assume the incident stokes vectors are already rotated to the BRDF reference frame.
    """

    def __init__(self,
                 is_training: bool,
                 data_dir: str,
                 dataset_type: str,
                 experiment_name: str,
                 batch_size: int,
                 max_epoch: int,
                 sample_num: int,
                 illum_sample_num: int,
                 positional_encoding_x_res: int,
                 positional_encoding_d_res: int,
                 gpu_num: int,
                 lr: float,
                 weight_dic: dict,
                 config_path: Path,
                 previous_stage_dir: str,
                 previous_stage_epoch_num: int,
                 stage_name: str,
                 max_step_ray_march: int,
                 use_mask: bool):

        super(TrainerNeISF, self).__init__(
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

        self.max_step_ray_march = max_step_ray_march
        self.stage_name = stage_name
        self.illum_sample_num = illum_sample_num

        self.embedder_dir = embedders.PEEmbedder(l_num=positional_encoding_d_res)

        # set up models.
        self.models_dic["incident_s0-net"] = neisf.IncidentNet(
            depth=4,
            width=256,
            in_ch=self.embedder_pos.out_dim + self.embedder_dir.out_dim,
            out_ch=3,  # 3-channel s0
            last_activation_func=nn.Softplus(),  # the value of the incident light can be greater than 1.0
            skips=[4]  # no skip
        )

        self.models_dic["incident_diff-net"] = neisf.IncidentNet(
            depth=4,
            width=256,
            in_ch=self.embedder_pos.out_dim + self.embedder_dir.out_dim,
            out_ch=3,  # 3-channel s1
            last_activation_func=nn.Tanh(),  # s1/s0 and s2/s0 are between [-1.0, 1.0]
            skips=[4]
        )

        self.models_dic["incident_spec-net"] = neisf.IncidentNet(
            depth=4,
            width=256,
            in_ch=self.embedder_pos.out_dim + self.embedder_dir.out_dim,
            out_ch=6,  # 3-channel s1, 3-channel s2
            last_activation_func=nn.Tanh(),  # s1/s0 and s2/s0 are between [-1.0, 1.0]
            skips=[4]
        )

        self.models_dic["rough-net"] = neisf.BRDFNet(
            depth=4,
            width=256,
            in_ch=self.embedder_pos.out_dim + self.shape_feature_vector_size,
            out_ch=1,  # roughness has 1 channel.
            skips=[4]
        )

        self.models_dic["albedo-net"] = neisf.BRDFNet(
            depth=4,
            width=256,
            in_ch=self.embedder_pos.out_dim + self.shape_feature_vector_size,
            out_ch=3,  # albedo has 3 channel.
            skips=[4]
        )

        if self.is_training:
            # load trained networks
            previous_stage_path = Path("results").joinpath(previous_stage_dir)

            if stage_name == "init":
                self.load_previous_training_models(previous_stage_path, previous_stage_epoch_num, ["sdfnet"])
                # fix SDF parameters during the init stage.
                for param in self.models_dic["sdfnet"].parameters():
                    param.requires_grad = False
            elif stage_name == "joint":
                self.load_previous_training_models(
                    previous_stage_path,
                    previous_stage_epoch_num,
                    ["sdfnet", "albedo-net", "rough-net", "incident_spec-net", "incident_diff-net", "incident_s0-net"],
                )
            else:
                raise NotImplementedError(f"stage must be in: ['init', 'joint'], yours: {stage_name}.")

            self.stokes_weight = weight_dic["stokes_weight"]
            self.loss_stokes = nn.L1Loss()

    def train_one_epoch(self, data_loader: DataLoader) -> float:
        loss_sum = 0

        prefetcher = DataPrefetcher(data_loader, self.device)

        for current_batch_num, batch in enumerate(prefetcher):
            s0 = batch["s0"]  # (b_size, 3)
            s1 = batch["s1"]  # (b_size, 3)
            s2 = batch["s2"]  # (b_size, 3)
            rays_o = batch["rays_o"]  # (b_size, 3)
            rays_d = batch["rays_d"]  # (b_size, 3)
            camera_axis = batch["camera_axis"]  # (b_size, 3)

            out_dic = self.forward(rays_o=rays_o, rays_d=rays_d, camera_axis=camera_axis, is_training=self.is_training)

            visibility = out_dic["visibility"].squeeze()  # (b_size,)

            dif_s0 = out_dic["dif_s0"]  # (b_size, 3)
            dif_s1 = out_dic["dif_s1"]  # (b_size, 3)
            dif_s2 = out_dic["dif_s2"]  # (b_size, 3)

            spe_s0 = out_dic["spe_s0"]  # (b_size, 3)
            spe_s1 = out_dic["spe_s1"]  # (b_size, 3)
            spe_s2 = out_dic["spe_s2"]  # (b_size, 3)

            z_samples_eik = out_dic["z_samples_eik"]

            # calc Eikonal loss.
            grad_theta = self.calc_grad_theta(rays_o[visibility], rays_d[visibility], z_samples_eik[visibility])
            ek_loss = self.loss_eikonal(grad_theta)

            s0_est = dif_s0 + spe_s0  # (b_size, 3)
            s1_est = dif_s1 + spe_s1  # (b_size, 3)
            s2_est = dif_s2 + spe_s2  # (b_size, 3)

            # calc Stokes loss.
            s0_loss = self.loss_stokes(s0_est[visibility], s0[visibility])
            s1_loss = self.loss_stokes(s1_est[visibility], s1[visibility])
            s2_loss = self.loss_stokes(s2_est[visibility], s2[visibility])
            loss = s0_loss + self.stokes_weight * (s1_loss + s2_loss) + self.eik_weight * ek_loss

            # update models.
            if self.stage_name == "init":
                self.update_all_models(loss)
            elif self.stage_name == "joint":
                self.update_all_models(loss, clip_sdf_grad=True, skip_nan=True)

            loss_sum += loss.item()

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
            w2c = batch_dic["w2c"][0][0].to(self.device)
            camera_axis_split = batch_dic["camera_axis"]  # (b_size, 3)

            out_srgb = []
            out_normal = []
            out_s0 = []
            out_s1 = []
            out_s2 = []
            out_diffuse = []
            out_specular = []
            out_rough = []
            out_albedo = []

            for rays_o, rays_d, camera_axis in zip(rays_o_split, rays_d_split, camera_axis_split):

                rays_o = rays_o.to(self.device)
                rays_d = rays_d.to(self.device)  # (b_size, 3)
                camera_axis = camera_axis.to(self.device)

                out_dic = self.forward(rays_o, rays_d, camera_axis, self.is_training)

                visibility = out_dic["visibility"]  # (b_size, 1)

                spe_s0 = out_dic["spe_s0"] * visibility  # (b_size, 3)
                spe_s1 = out_dic["spe_s1"] * visibility  # (b_size, 3)
                spe_s2 = out_dic["spe_s2"] * visibility  # (b_size, 3)

                dif_s0 = out_dic["dif_s0"] * visibility  # (b_size, 3)
                dif_s1 = out_dic["dif_s1"] * visibility  # (b_size, 3)
                dif_s2 = out_dic["dif_s2"] * visibility  # (b_size, 3)

                normals = out_dic["normal"]  # (b_size, 3)
                rough = out_dic["roughness"]  # (b_size, 1)
                albedo = out_dic["albedo"]  # (b_size, 1)

                # convert normal to [0., 1.]
                normals = polarutils.rotate_normal_from_world_to_tangent(w2c=w2c, rays_d=rays_d, normals=normals)
                normals = (normals + 1.) / 2.

                # compute stokes vectors
                s0 = dif_s0 + spe_s0
                s1 = dif_s1 + spe_s1
                s2 = dif_s2 + spe_s2

                srgb = gamma(s0)

                # store
                out_srgb.append(srgb.detach().cpu())
                out_normal.append(normals.detach().cpu())
                out_s0.append(s0.detach().cpu())
                out_s1.append(s1.detach().cpu())
                out_s2.append(s2.detach().cpu())
                out_diffuse.append(dif_s0.detach().cpu())
                out_specular.append(spe_s0.detach().cpu())
                out_rough.append(rough.detach().cpu())
                out_albedo.append(albedo.detach().cpu())

            out_srgb = torch.cat(out_srgb, dim=0)
            out_normal = torch.cat(out_normal, dim=0)
            out_s0 = torch.cat(out_s0, dim=0)
            out_s1 = torch.cat(out_s1, dim=0)
            out_s2 = torch.cat(out_s2, dim=0)
            out_diffuse = torch.cat(out_diffuse, dim=0)
            out_specular = torch.cat(out_specular, dim=0)
            out_rough = torch.cat(out_rough, dim=0)
            out_albedo = torch.cat(out_albedo, dim=0)

            if self.use_mask:
                mask = self.data.mask_pool[img_idx]
            else:
                mask = None

            images_dic = self.reshape_and_scale_all_images(
                valid_mask=mask, max_val=None,
                srgb=out_srgb, diffuse=out_diffuse, specular=out_specular, albedo=out_albedo,
                normal=out_normal, roughness=out_rough, s0=out_s0, s1=out_s1, s2=out_s2,
            )
            _write_all_images_in_dict_for_neisf(save_path, img_idx, images_dic)

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, camera_axis: torch.Tensor, is_training: bool) -> dict:
        # sampling the positions along the rays.
        z_vals, z_samples_eik = self.sampler.get_z_vals(rays_o, rays_d, self.models_dic["sdfnet"], is_training)

        # compute sampled 3D positions along the ray.
        rays_d_usq = rays_d.unsqueeze(1)  # (b, 1, 3)
        pos_flat = (rays_o.unsqueeze(1) + rays_d_usq * z_vals.unsqueeze(2)).reshape(-1, 3)  # (bz, 3)

        # input the positions to SDFNet.
        sdfv_flat, fvec_flat, grad_flat = \
            self.models_dic["sdfnet"].get_all_outputs(pos_flat, self.sampler.bounding_sphere_r, is_training)

        # volume rendering.
        weights = renderutils.volume_rendering(z_vals, self.models_dic["sdfnet"], sdfv_flat, self.device)  # (b, z)

        # compute normals
        normal = calc_weighted_sum(weights=weights, target=normalize_tensor(grad_flat))  # (b, 3)
        normal = normalize_tensor(normal)  # (b, 3)

        # compute ray-object interation points
        with torch.no_grad():
            t_val = ray_march(rays_o, rays_d, self.models_dic["sdfnet"], self.max_step_ray_march)  # (b, 3)
            t_val = torch.clip(t_val, min=0.)
            pos = rays_o + rays_d * t_val  # (b, 3)

        # embed interaction position pos and sampling 3D position pos_flat
        pos_embed = self.embedder_pos.embed(pos)  # (b, pe.out_dim)
        pos_embed_s = pos_embed.unsqueeze(1).repeat(1, self.illum_sample_num, 1).reshape(-1,
                                                                                         self.embedder_pos.out_dim)  # (bs, pe.out_dim)
        pos_embed_flat = self.embedder_pos.embed(pos_flat)  # (bz, pe.out_dim)

        # embed incident direction
        wi_s = renderutils.fibonacci_sphere_sampling(normal, self.illum_sample_num, self.device).reshape(-1, 3)  # (bs, 3)
        wi_embed_s = self.embedder_dir.embed(wi_s)  # (bs, de.out_dim)

        # ------------MLP-------------
        # inputs with or without incident directions
        inp_w_dir = torch.cat([pos_embed_s, wi_embed_s], dim=1)  # (bs, pe.outdim + de.outdim)
        inp_wo_dir = torch.cat([pos_embed_flat, fvec_flat], dim=1)  # (bz, pe.outdim + fv_size)

        # estimate incident s0
        incident_s0 = self.models_dic["incident_s0-net"](inp_w_dir)  # (bs, 3)

        # estimate incident specular s1 and s2
        incident_spec = self.models_dic["incident_spec-net"](inp_w_dir)  # (bs, 6)
        incident_s1_spec = incident_spec[:, :3] * incident_s0  # (bs, 3)
        incident_s2_spec = incident_spec[:, 3:] * incident_s0  # (bs, 3)

        # estimate incident diffuse s1
        incident_s1_diff = self.models_dic["incident_diff-net"](inp_w_dir)  # (bs, 3)
        incident_s1_diff = incident_s1_diff * incident_s0  # (bs, 3)

        # estimate roughness.
        rough_flat = self.models_dic["rough-net"](inp_wo_dir)  # (bz, 1)
        rough = calc_weighted_sum(weights=weights, target=rough_flat)  # (b, 1)
        rough = rough * 0.5 + 0.5  # make sure roughness is between [0, 1]  (b, 1)
        rough_s = rough.unsqueeze(1).repeat(1, self.illum_sample_num, 1).reshape(-1, 1)  # (bs, 1)

        # estimate albedo.
        albedo_flat = self.models_dic["albedo-net"](inp_wo_dir)  # (bz, 3)
        albedo = calc_weighted_sum(weights=weights, target=albedo_flat)  # (b, 3)
        albedo = albedo * 0.5 + 0.5  # make sure albedo is between [0, 1]  (b, 3)
        albedo_s = albedo.unsqueeze(1).repeat(1, self.illum_sample_num, 1).reshape(-1, 3)  # (bs, 3)

        # ------------Rendering-------------
        # repeat incident ray direction and surface normal
        rays_d_s = rays_d.unsqueeze(1).repeat(1, self.illum_sample_num, 1).reshape(-1, 3)  # (bs, 3)
        normal_s = normal.unsqueeze(1).repeat(1, self.illum_sample_num, 1).reshape(-1, 3)  # (bs, 3)
        camera_axis_s = camera_axis.unsqueeze(1).repeat(1, self.illum_sample_num, 1).reshape(-1, 3)  # (bs, 3)

        # compute cos values
        ndl_s = dot_product_tensor(normal_s, wi_s)  # (bs, 1)

        # set the contribution of the backside surface to 0.
        with torch.no_grad():
            ndo = dot_product_tensor(normal, -rays_d)  # (bs, 1)
            visibility = ndo > 0.  # (bs, 1)

        # render the Distribution and Geometry term
        fs_s = brdfutils.calc_ggx_reflectance_baek_no_fresnel(
            view_dir=-rays_d_s, light_dir=wi_s, normal=normal_s, roughness=rough_s,
        )  # (bs, 1)

        # render Stokes
        dif_stokes_s = polarutils.calc_diffuse_stokes_full(
            wo_wld=-rays_d_s,
            wi_wld=wi_s,
            n_wld=normal_s,
            cam_axis=camera_axis_s,
            s0=incident_s0,
            s1=incident_s1_diff
        )
        dif_s0_s, dif_s1_s, dif_s2_s = dif_stokes_s

        spe_stokes_s = polarutils.calc_specular_stokes_full(
            wo_wld=-rays_d_s,
            wi_wld=wi_s,
            cam_axis=camera_axis_s,
            s0=incident_s0,
            s1=incident_s1_spec,
            s2=incident_s2_spec
        )
        spe_s0_s, spe_s1_s, spe_s2_s = spe_stokes_s

        i_spe_s = fs_s * ndl_s * renderutils.ILLUM_AREA  # (bs, 3)
        i_dif_s = albedo_s / torch.pi * ndl_s * renderutils.ILLUM_AREA  # (bs, 3)

        spe_s0_s = i_spe_s * spe_s0_s  # (bs, 3)
        spe_s1_s = i_spe_s * spe_s1_s  # (bs, 3)
        spe_s2_s = i_spe_s * spe_s2_s  # (bs, 3)

        dif_s0_s = i_dif_s * dif_s0_s  # (bs, 3)
        dif_s1_s = i_dif_s * dif_s1_s  # (bs, 3)
        dif_s2_s = i_dif_s * dif_s2_s  # (bs, 3)

        spe_s0 = torch.mean(spe_s0_s.reshape(-1, self.illum_sample_num, 3), dim=1)  # (b, 3)
        spe_s1 = torch.mean(spe_s1_s.reshape(-1, self.illum_sample_num, 3), dim=1)  # (b, 3)
        spe_s2 = torch.mean(spe_s2_s.reshape(-1, self.illum_sample_num, 3), dim=1)  # (b, 3)

        dif_s0 = torch.mean(dif_s0_s.reshape(-1, self.illum_sample_num, 3), dim=1)  # (b, 3)
        dif_s1 = torch.mean(dif_s1_s.reshape(-1, self.illum_sample_num, 3), dim=1)  # (b, 3)
        dif_s2 = torch.mean(dif_s2_s.reshape(-1, self.illum_sample_num, 3), dim=1)  # (b, 3)

        out_dic = {
            "dif_s0": dif_s0,
            "dif_s1": dif_s1,
            "dif_s2": dif_s2,
            "spe_s0": spe_s0,
            "spe_s1": spe_s1,
            "spe_s2": spe_s2,
            "normal": normal,
            "visibility": visibility,
            "roughness": rough,
            "albedo": albedo,
            "z_samples_eik": z_samples_eik
        }
        return out_dic

    def inference_uv(self,
                     epoch_num: int,
                     mesh_path: Path,
                     uv_height: int,
                     uv_width: int,
                     albedo_path: Path,
                     roughness_path: Path,
                     mtl_path: Path) -> None:
        """ export uv texture maps of roughness and diffuse albedo.

        Args:
            epoch_num: epoch number.
            mesh_path: file path to the target mesh.
            uv_height: height of uv texture.
            uv_width: width of uv texture.
            albedo_path: file path to save albedo.
            roughness_path: file path to save roughness.
            mtl_path: file path to save .mtl file.
        """

        self.locate_all_models_to_device()
        self.load_all_models(epoch_num)
        self.set_mode_all_models(is_training=False)

        v, tc, _, f, ftc, _ = igl.read_obj(str(mesh_path), dtype="float32")

        if len(tc) == 0:
            raise ValueError("your obj file does not contain texture coordinate.")

        texture_image = np.zeros((uv_height, uv_width, 4), dtype=np.float32)  # albedo x 3-ch, roughness x 1-ch.
        weight_image = np.zeros((uv_height, uv_width), dtype=np.float32)

        # compute the color five times.
        # refer to: https://github.com/Kai-46/IRON/blob/main/models/export_materials.py#L174.
        for i in range(5):
            # compute sample points.
            pts, uv_pts = shapeutils.sample_surface(v, f, tc, ftc)  # (s, 3), (s, 2).
            pts = torch.from_numpy(pts)

            merge_texture = []

            for pts_split in torch.split(pts, self.batch_size):
                pts_split = pts_split.to(self.device)  # (b_size, 3).

                with torch.no_grad():
                    # input the positions to SDFNet.
                    _, fvec, grad = self.models_dic["sdfnet"].get_all_outputs(
                        pts_split, self.sampler.bounding_sphere_r, is_training=False)

                    pts_embed = self.embedder_pos.embed(pts_split)
                    inp_wo_dir = torch.cat([pts_embed, fvec], dim=1)  # (b, pe.outdim + fv_size)

                    # estimate albedo.
                    albedo = self.models_dic["albedo-net"](inp_wo_dir)  # (bz, 3)
                    albedo = albedo * 0.5 + 0.5  # make sure albedo is between [0, 1].

                    # estimate roughness.
                    rough = self.models_dic["rough-net"](inp_wo_dir)  # (b, 1)
                    rough = rough * 0.5 + 0.5  # make sure roughness is between [0, 1].

                    merge_texture.append(torch.cat((albedo, rough), dim=-1).detach().cpu())

            merge_texture = torch.cat(merge_texture, dim=0).numpy()

            texture_image, weight_image = shapeutils.accumulate_splat_material(
                texture_image=texture_image, weight_image=weight_image, uv=uv_pts, merged_texture=merge_texture,
            )

        final_texture_image = texture_image / (weight_image[:, :, np.newaxis] + 1e-10)
        final_texture_image = np.clip(final_texture_image * MAX_8BIT, 0, MAX_8BIT)

        final_diffuse_image = final_texture_image[:, :, :3]
        final_roughness_image = np.repeat(final_texture_image[:, :, 3:], 3, axis=2)

        my_write_image_8bit(albedo_path, final_diffuse_image)
        my_write_image_8bit(roughness_path, final_roughness_image)

        with open(mtl_path, "w") as fp:
            fp.write(
                "newmtl Wood\n"
                "Ka 1.000000 1.000000 1.000000\n"
                "Kd 0.640000 0.640000 0.640000\n"
                "Ks 0.500000 0.500000 0.500000\n"
                "Ns 96.078431\n"
                "Ni 1.000000\n"
                "d 1.000000\n"
                "illum 0\n"
                f"map_Kd {albedo_path.name}\n"
            )

    def render_relighting(self, image_dir_name: str, epoch_num: int, env_map_name: str, light_num: int) -> None:
        """ Render relighting images.

        Args:
            image_dir_name: the name of image folder.
            epoch_num: epoch number to be used for rendering.
            env_map_name: the name of environment map.
            light_num: the number of sample illumination.
        """

        env_max = 50.
        env_gain = 1.  # needs to be moved to outside the function as global variables.

        # setup dataset.
        self.data = get_dataset(self.dataset_type, self.data_dir, use_mask=False)

        # create save folder.
        if not self.out_dir_path.exists():
            raise FileNotFoundError(f"Cannot find your result folder: {self.out_dir_path}")
        save_path = self.out_dir_path.joinpath(f"images_ep{epoch_num:05d}/re_lighting_{image_dir_name}_{env_map_name}")
        save_path.mkdir(parents=True)

        # prepare trained models.
        self.locate_all_models_to_device()
        self.load_all_models(epoch_num)
        self.set_mode_all_models(False)

        # prepare env map.
        env_map_path = Path(ENV_MAP_FOLDER_NAME).joinpath(f"{env_map_name}.exr")
        img_env = my_read_image(env_map_path)

        h_env, w_env = img_env.shape[:2]

        img_env = torch.clip(torch.from_numpy(img_env * env_gain).to(self.device), 0., env_max)
        img_env = img_env.reshape(-1, 3)  # (h_env * w_env, 3)

        # main.
        with torch.no_grad():
            self.render_relighting_sub(save_path, img_env, h_env, w_env, light_num)

    def render_relighting_sub(self,
                              save_path: Path,
                              env_map: torch.Tensor,
                              h_env: int,
                              w_env: int,
                              light_num: int) -> None:
        """ See RelightingMixin.render_relighting_sub() for more detailed description. """

        for frame in range(self.data.n_images):
            batch_dic = self.data.getitem_by_frame_id_split(frame, split_size=self.batch_size)

            result_diffuse = []
            result_specular = []

            for ii in range(len(batch_dic["s0"])):
                rays_o = batch_dic["rays_o"][ii].to(self.device)
                rays_d = batch_dic["rays_d"][ii].to(self.device)
                camera_axis = batch_dic["camera_axis"][ii].to(self.device)

                mask = batch_dic["mask"][ii]
                if torch.sum(mask) == 0:  # no valid pixels.
                    dif_s0 = torch.zeros(self.batch_size, 3)
                    spe_s0 = torch.zeros(self.batch_size, 3)
                else:
                    out_dic = self.forward(rays_o, rays_d, camera_axis, is_training=False)

                    normal_wld = out_dic["normal"]  # (b_size, 3)
                    rough = out_dic["roughness"]  # (b_size, 1)
                    albedo = out_dic["albedo"]  # (b_size, 3)

                    # sample illumination.
                    wi = renderutils.fibonacci_sphere_sampling(normal_wld, light_num, self.device)  # (b, s, 3)
                    illum = renderutils.sample_illum_from_env(env_map, wi, h_env, w_env)  # (b, s, 3)

                    # change the shape of tensors for rendering.
                    rays_o_bs = rays_o.unsqueeze(1).repeat(1, light_num, 1).reshape(-1, 3)  # (bs, 3)
                    rays_d_bs = rays_d.unsqueeze(1).repeat(1, light_num, 1).reshape(-1, 3)  # (bs, 3)
                    camera_axis_bs = camera_axis.unsqueeze(1).repeat(1, light_num, 1).reshape(-1, 3)  # (bs, 3)
                    normal_wld_bs = normal_wld.unsqueeze(1).repeat(1, light_num, 1).reshape(-1, 3)  # (bs, 3)
                    wi_bs = wi.reshape(-1, 3)  # (bs, 3)
                    illum_bs = illum.reshape(-1, 3)  # (bs, 3)
                    rough_bs = rough.unsqueeze(1).repeat(1, light_num, 1).reshape(-1, 1)  # (bs, 3)
                    albedo_bs = albedo.unsqueeze(1).repeat(1, light_num, 1).reshape(-1, 3)  # (bs, 3)
                    dummy_stokes_bs = torch.zeros_like(rays_o_bs)  # (bs, 3)

                    ndl_bs = dot_product_tensor(normal_wld_bs, wi_bs)  # (bs, 1)
                    ndo_bs = dot_product_tensor(normal_wld_bs, -rays_d_bs)  # (bs, 1)
                    visibility_bs = brdfutils.calc_chi(ndo_bs)  # (bs, 1)

                    # render specular reflectance (bs, 1).
                    fs_bs = brdfutils.calc_ggx_reflectance_baek_no_fresnel(
                        view_dir=-rays_d_bs, light_dir=wi_bs, normal=normal_wld_bs, roughness=rough_bs,
                    )

                    # render Stokes.
                    dif_stokes_s = polarutils.calc_diffuse_stokes_full(
                        wo_wld=-rays_d_bs,
                        wi_wld=wi_bs,
                        n_wld=normal_wld_bs,
                        cam_axis=camera_axis_bs,
                        s0=illum_bs,
                        s1=dummy_stokes_bs
                    )
                    spe_stokes_s = polarutils.calc_specular_stokes_full(
                        wo_wld=-rays_d_bs,
                        wi_wld=wi_bs,
                        cam_axis=camera_axis_bs,
                        s0=illum_bs,
                        s1=dummy_stokes_bs,
                        s2=dummy_stokes_bs
                    )
                    dif_s0_s, dif_s1_s, dif_s2_s = dif_stokes_s
                    spe_s0_s, spe_s1_s, spe_s2_s = spe_stokes_s

                    i_spe_bs = fs_bs * ndl_bs * renderutils.ILLUM_AREA * visibility_bs  # (bs, 3)
                    i_dif_bs = albedo_bs / torch.pi * ndl_bs * renderutils.ILLUM_AREA * visibility_bs  # (bs, 3)

                    spe_s0_s = i_spe_bs * spe_s0_s  # (bs, 3)
                    dif_s0_s = i_dif_bs * dif_s0_s  # (bs, 3)

                    spe_s0 = torch.mean(spe_s0_s.reshape(-1, light_num, 3), dim=1)  # (b, 3)
                    dif_s0 = torch.mean(dif_s0_s.reshape(-1, light_num, 3), dim=1)  # (b, 3)

                result_diffuse.append(dif_s0.detach().cpu())
                result_specular.append(spe_s0.detach().cpu())

            result_diffuse = torch.cat(result_diffuse, dim=0)
            result_specular = torch.cat(result_specular, dim=0)

            result_diffuse = torch.clip(result_diffuse, 0., 1.)
            result_specular = torch.clip(result_specular, 0., 1.)
            result_rgb = torch.clip(result_diffuse + result_specular, 0., 1.)

            images_dic = self.reshape_and_scale_all_images(
                valid_mask=self.data.mask_pool[frame], max_val=1,
                rgb=result_rgb, diffuse=result_diffuse, specular=result_specular,
            )
            for image_name in images_dic:
                my_write_image_exr(
                    save_path.joinpath(f"{frame + 1: 03d}_{image_name}.exr"), images_dic[image_name].numpy()
                )


class TrainerNeISFNoStokes(TrainerBase):
    """ The unpolarized version Neural Incident Stokes Field (NeISF).

    This trainer has four MLPs for SDF, incident_s0, albedo, and roughness.
    images are rendered using a PBR

    Attributes:
        illum_sample_num (int): the number of samples for the incident light.
        loss_stokes (torch.nn.modules.loss): a loss function for stokes values.
        max_step_ray_march (int): the number of steps for the ray marching.
    """

    def __init__(self,
                 is_training: bool,
                 data_dir: str,
                 dataset_type: str,
                 experiment_name: str,
                 batch_size: int,
                 max_epoch: int,
                 sample_num: int,
                 illum_sample_num: int,
                 positional_encoding_x_res: int,
                 positional_encoding_d_res: int,
                 gpu_num: int,
                 lr: float,
                 weight_dic: dict,
                 config_path: Path,
                 previous_stage_dir: str,
                 previous_stage_epoch_num: int,
                 stage_name: str,
                 max_step_ray_march: int,
                 use_mask: bool):

        super(TrainerNeISFNoStokes, self).__init__(
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
            use_mask=use_mask
        )

        self.max_step_ray_march = max_step_ray_march  # set the steps of ray marching
        self.stage_name = stage_name
        self.illum_sample_num = illum_sample_num

        self.embedder_dir = embedders.PEEmbedder(l_num=positional_encoding_d_res)

        # set up models.
        self.models_dic["incident_s0-net"] = neisf.IncidentNet(
            depth=4,
            width=256,
            in_ch=self.embedder_pos.out_dim + self.embedder_dir.out_dim,
            out_ch=3,  # 3-channel s0
            last_activation_func=nn.Softplus(),  # the value of the incident light can be greater than 1.0
            skips=[4]  # no skip
        )

        self.models_dic["rough-net"] = neisf.BRDFNet(
            depth=4,
            width=256,
            in_ch=self.embedder_pos.out_dim + self.shape_feature_vector_size,
            out_ch=1,  # roughness has 1 channel.
            skips=[4]
        )

        self.models_dic["albedo-net"] = neisf.BRDFNet(
            depth=4,
            width=256,
            in_ch=self.embedder_pos.out_dim + self.shape_feature_vector_size,
            out_ch=3,  # albedo has 3 channel.
            skips=[4]
        )

        if self.is_training:
            # load trained networks
            previous_stage_path = Path("results").joinpath(previous_stage_dir)
            if stage_name == "init":
                self.load_previous_training_models(previous_stage_path, previous_stage_epoch_num, ["sdfnet"])
                # fix SDF parameters during the init stage.
                for param in self.models_dic["sdfnet"].parameters():
                    param.requires_grad = False
            elif stage_name == "joint":
                self.load_previous_training_models(
                    previous_stage_path,
                    previous_stage_epoch_num,
                    ["sdfnet", "albedo-net", "rough-net", "incident_s0-net"],
                )
            else:
                raise NotImplementedError(f"stage must be in: ['init', 'joint'], yours: {stage_name}.")

        self.loss_stokes = nn.L1Loss()

    def train_one_epoch(self, data_loader: DataLoader) -> float:
        loss_sum = 0

        prefetcher = DataPrefetcher(data_loader, self.device)

        for current_batch_num, batch in enumerate(prefetcher):
            s0 = batch["s0"]  # (b_size, 3)
            rays_o = batch["rays_o"]  # (b_size, 3)
            rays_d = batch["rays_d"]  # (b_size, 3)

            out_dic = self.forward(rays_o=rays_o, rays_d=rays_d, is_training=self.is_training)

            dif_s0 = out_dic["dif_s0"]  # (b_size, 3)
            spe_s0 = out_dic["spe_s0"]  # (b_size, 3)
            visibility = out_dic["visibility"].squeeze()  # (b_size,)
            z_samples_eik = out_dic["z_samples_eik"]

            # Calc Eikonal loss.
            grad_theta = self.calc_grad_theta(rays_o[visibility], rays_d[visibility], z_samples_eik[visibility])
            ek_loss = self.loss_eikonal(grad_theta)

            s0_est = dif_s0 + spe_s0  # (b_size, 3)

            # Calc RGB loss.
            s0_loss = self.loss_stokes(s0_est[visibility], s0[visibility])
            loss = s0_loss + self.eik_weight * ek_loss

            # update models.
            if self.stage_name == "init":
                self.update_all_models(loss)
            elif self.stage_name == "joint":
                self.update_all_models(loss, clip_sdf_grad=True, skip_nan=True)

            loss_sum += loss.item()

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
            w2c = batch_dic["w2c"][0][0].to(self.device)

            out_srgb = []
            out_normal = []
            out_s0 = []
            out_diffuse = []
            out_specular = []
            out_rough = []
            out_albedo = []

            for rays_o, rays_d in zip(rays_o_split, rays_d_split):

                rays_o = rays_o.to(self.device)
                rays_d = rays_d.to(self.device)  # (b_size, 3)

                out_dic = self.forward(rays_o, rays_d, self.is_training)

                visibility = out_dic["visibility"]  # (b_size, 1)

                spe_s0 = out_dic["spe_s0"] * visibility  # (b_size, 3)
                dif_s0 = out_dic["dif_s0"] * visibility  # (b_size, 3)

                normals = out_dic["normal"]  # (b_size, 3)
                rough = out_dic["roughness"]  # (b_size, 1)
                albedo = out_dic["albedo"]  # (b_size, 1)

                # convert normal to [0., 1.]
                normals = polarutils.rotate_normal_from_world_to_tangent(w2c=w2c, rays_d=rays_d, normals=normals)
                normals = (normals + 1.) / 2.

                # compute stokes vectors
                s0 = dif_s0 + spe_s0
                srgb = gamma(s0)

                # store
                out_srgb.append(srgb.detach().cpu())
                out_normal.append(normals.detach().cpu())
                out_s0.append(s0.detach().cpu())
                out_diffuse.append(dif_s0.detach().cpu())
                out_specular.append(spe_s0.detach().cpu())
                out_rough.append(rough.detach().cpu())
                out_albedo.append(albedo.detach().cpu())

            out_srgb = torch.cat(out_srgb, dim=0)
            out_normal = torch.cat(out_normal, dim=0)
            out_s0 = torch.cat(out_s0, dim=0)
            out_diffuse = torch.cat(out_diffuse, dim=0)
            out_specular = torch.cat(out_specular, dim=0)
            out_rough = torch.cat(out_rough, dim=0)
            out_albedo = torch.cat(out_albedo, dim=0)

            if self.use_mask:
                mask = self.data.mask_pool[img_idx]
            else:
                mask = None

            images_dic = self.reshape_and_scale_all_images(
                valid_mask=mask, max_val=None,
                rgb=out_srgb, diffuse=out_diffuse, specular=out_specular, albedo=out_albedo,
                normal=out_normal, roughness=out_rough, s0=out_s0,
            )
            _write_all_images_in_dict_for_neisf(save_path, img_idx, images_dic)

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, is_training: bool) -> dict:
        # sampling the positions along the rays.
        z_vals, z_samples_eik = self.sampler.get_z_vals(rays_o, rays_d, self.models_dic["sdfnet"], is_training)

        # compute sampled 3D positions along the ray.
        rays_d_usq = rays_d.unsqueeze(1)  # (b, 1, 3)
        pos_flat = (rays_o.unsqueeze(1) + rays_d_usq * z_vals.unsqueeze(2)).reshape(-1, 3)  # (bz, 3)

        # input the positions to SDFNet.
        sdfv_flat, fvec_flat, grad_flat = \
            self.models_dic["sdfnet"].get_all_outputs(pos_flat, self.sampler.bounding_sphere_r, is_training)

        # volume rendering.
        weights = renderutils.volume_rendering(z_vals, self.models_dic["sdfnet"], sdfv_flat, self.device)  # (b, z)

        # compute normals
        normal = calc_weighted_sum(weights=weights, target=normalize_tensor(grad_flat))  # (b, 3)
        normal = normalize_tensor(normal)  # (b, 3)

        # compute ray-object interation points
        with torch.no_grad():
            t_val = ray_march(rays_o, rays_d, self.models_dic["sdfnet"], self.max_step_ray_march)  # (b, 3)
            t_val = torch.clip(t_val, min=0.)
            pos = rays_o + rays_d * t_val  # (b, 3)

        # embed interaction position pos and sampling 3D position pos_flat
        pos_embed = self.embedder_pos.embed(pos)  # (b, pe.out_dim)
        pos_embed_s = pos_embed.unsqueeze(1).repeat(1, self.illum_sample_num, 1).reshape(-1, self.embedder_pos.out_dim)  # (bs, pe.out_dim)
        pos_embed_flat = self.embedder_pos.embed(pos_flat)  # (bz, pe.out_dim)

        # embed incident direction
        wi_s = renderutils.fibonacci_sphere_sampling(normal, self.illum_sample_num, self.device).reshape(-1, 3)  # (bs, 3)
        wi_embed_s = self.embedder_dir.embed(wi_s)  # (bs, de.out_dim)

        # ------------MLP-------------
        # inputs with or without incident directions
        inp_w_dir = torch.cat([pos_embed_s, wi_embed_s], dim=1)  # (bs, pe.outdim + de.outdim)
        inp_wo_dir = torch.cat([pos_embed_flat, fvec_flat], dim=1)  # (bz, pe.outdim + fv_size)

        # estimate incident s0
        incident_s0 = self.models_dic["incident_s0-net"](inp_w_dir)  # (bs, 3)

        # estimate roughness.
        rough_flat = self.models_dic["rough-net"](inp_wo_dir)  # (bz, 1)
        rough = calc_weighted_sum(weights=weights, target=rough_flat)  # (b, 1)
        rough = rough * 0.5 + 0.5  # make sure roughness is between [0, 1]  (b, 1)
        rough_s = rough.unsqueeze(1).repeat(1, self.illum_sample_num, 1).reshape(-1, 1)  # (bs, 1)

        # estimate albedo.
        albedo_flat = self.models_dic["albedo-net"](inp_wo_dir)  # (bz, 3)
        albedo = calc_weighted_sum(weights=weights, target=albedo_flat)  # (b, 3)
        albedo = albedo * 0.5 + 0.5  # make sure albedo is between [0, 1]  (b, 3)
        albedo_s = albedo.unsqueeze(1).repeat(1, self.illum_sample_num, 1).reshape(-1, 3)  # (bs, 3)

        # ------------Rendering-------------
        # repeat incident ray direction and surface normal
        rays_d_s = rays_d.unsqueeze(1).repeat(1, self.illum_sample_num, 1).reshape(-1, 3)  # (bs, 3)
        normal_s = normal.unsqueeze(1).repeat(1, self.illum_sample_num, 1).reshape(-1, 3)  # (bs, 3)

        # compute cos values
        ndl_s = dot_product_tensor(normal_s, wi_s)  # (bs, 1)

        # set the contribution of the backside surface to 0.
        with torch.no_grad():
            ndo = dot_product_tensor(normal, -rays_d)  # (bs, 1)
            visibility = ndo > 0.  # (bs, 1)

        # render the Distribution and Geometry term
        fs_s = brdfutils.calc_ggx_reflectance_baek_no_fresnel(
            view_dir=-rays_d_s, light_dir=wi_s, normal=normal_s, roughness=rough_s,
        )  # (bs, 1)

        # render Fresnel term
        dif_fresnel = polarutils.calc_diffuse_fresnel(wo_wld=-rays_d_s, wi_wld=wi_s, n_wld=normal_s)  # (bs, 1)
        spe_fresnel = polarutils.calc_specular_fresnel(wo_wld=-rays_d_s, wi_wld=wi_s)  # (bs, 1)

        # rendering equation
        spe_s0_s = fs_s * ndl_s * spe_fresnel * incident_s0 * renderutils.ILLUM_AREA  # (bs, 3)
        dif_s0_s = albedo_s / torch.pi * ndl_s * dif_fresnel * incident_s0 * renderutils.ILLUM_AREA  # (bs, 3)

        spe_s0 = torch.mean(spe_s0_s.reshape(-1, self.illum_sample_num, 3), dim=1)  # (b, 3)
        dif_s0 = torch.mean(dif_s0_s.reshape(-1, self.illum_sample_num, 3), dim=1)  # (b, 3)

        out_dic = {
            "dif_s0": dif_s0,
            "spe_s0": spe_s0,
            "normal": normal,
            "roughness": rough,
            "albedo": albedo,
            "visibility": visibility,
            "z_samples_eik": z_samples_eik
        }
        return out_dic
