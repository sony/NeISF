# trainers_base.py
""" This script defines some base trainer classes that will be inherited by other trainer classes.

Classes:
    - TrainerBase: Base class of all the other trainers. Include attributes and methods that are necessary for all
        the other trainers. Note that SDFNet is already defined for avoiding some warnings. And also note that loss
        functions are not included except the Eikonal loss.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from pathlib import Path
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from .utils import assign_device

import mymodules.models.volsdf as volsdf

from mymodules import blenderutils as bl
from mymodules import embedders
from mymodules import shapeutils
from mymodules.datasets import get_dataset
from mymodules.globals import RESULT_PARENT_PATH, BOUNDING_SPHERE_R
from mymodules.losses import EikonalLoss
from mymodules.samplers.volsdfsampler import ErrorBoundPosSampler

MODELS_FOLDER_NAME = "models"
SAVED_CONFIG_NAME = "parameters.json"


class TrainerBase:
    """ Base class of all the other trainers.

    Attributes:
        batch_size (int): the number of pixels sampled in each iteration.
        config_path (Path): file path of config file.
        current_epoch (int): the number of current epoch.
        data (Dataset): dataset class. For more details, see `mymodules/datasets.py`.
        data_dir (str): directory name of your target data.
        data_loader (DataLoader): data loader. See 'https://pytorch.org/docs/stable/data.html'.
        device (torch.device): device for training.
        eik_weight (float): a weight used for computing the Eikonal loss.
        embedder_pos (Embedder): positional encoding for 3D positions.
        is_training (bool): True when training, False otherwise.
        loss_eikonal (EikonalLoss): eikonal loss.
        lr (float): learning rate.
        max_epoch (int): maximum epochs.
        models_dic (dict): this dict contains all the models to be trained. {model_name (str): models}.
        optims_dic (dict): optimizers of the models in models_dic.
        out_dir_path (Path): file path of result folder.
        sample_num (int): the number of 3D positions sampled along each ray.
        sampler (ErrorBoundPosSampler): SDF-based 3D position sampler introduced in volSDF.
        scheduler_dic (dict): schedulers for weight decay.
        shape_feature_vector_size (int): the size of feature vector size from SDF-net.
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
                 gpu_num: int,
                 lr: float,
                 weight_dic: dict,
                 config_path: Path,
                 use_mask: bool):

        self.is_training = is_training
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.sample_num = sample_num
        self.device = assign_device(gpu_num)
        self.lr = lr
        self.config_path = config_path
        self.out_dir_path = RESULT_PARENT_PATH.joinpath(experiment_name)
        self.use_mask = use_mask
        self.dataset_type = dataset_type

        # load dataset.
        self.data = None
        self.data_loader = None
        self.batch_num = None

        self.models_dic = {}
        self.optims_dic = {}
        self.scheduler_dic = {}

        self.current_epoch = 1

        self.embedder_pos = embedders.PEEmbedder(l_num=positional_encoding_x_res)

        # set up sampler.
        self.sampler = ErrorBoundPosSampler(
            near=0.,
            far=2 * BOUNDING_SPHERE_R,
            n_samples=sample_num,
            n_samples_eval=128,
            n_samples_extra=32,
            bounding_sphere_r=BOUNDING_SPHERE_R,
            device=self.device,
        )

        self.shape_feature_vector_size = 256

        self.models_dic["sdfnet"] = volsdf.SDFNet(
            depth=8,
            width=256,
            in_ch=self.embedder_pos.out_dim,
            out_ch=1 + self.shape_feature_vector_size,
            skips=[4],
            embedder=self.embedder_pos,
            sphere_scale=1.0,
        )

        self.eik_weight = weight_dic["eik_weight"]
        self.loss_eikonal = EikonalLoss()

    def train(self):
        """ Main function for training.
        Your trainer must implement `train_one_epoch` function.
        """

        assert self.is_training is True

        # prepare output folder.
        self.out_dir_path.joinpath(MODELS_FOLDER_NAME).mkdir(parents=True)
        shutil.copy(self.config_path, self.out_dir_path.joinpath(SAVED_CONFIG_NAME))

        # prepare dataset.
        self.data = get_dataset(dataset_type=self.dataset_type, data_dir=self.data_dir, use_mask=self.use_mask)
        self.data_loader = DataLoader(
            self.data, shuffle=True, batch_size=self.batch_size, pin_memory=True, drop_last=True, num_workers=1,
        )
        self.batch_num = len(self.data_loader)

        # prepare models.
        self.locate_all_models_to_device()
        self.set_all_optimizers()
        self.set_all_schedulers(decay_steps=self.max_epoch * self.batch_num)

        # main iteration.
        for epoch in range(self.max_epoch):
            self.set_mode_all_models(is_training=self.is_training)

            f = open(self.out_dir_path.joinpath("loss.txt"), "a")

            loss_sum_epoch = self.train_one_epoch(self.data_loader)

            f.write(f"epoch:{self.current_epoch:05d}  train_loss:{loss_sum_epoch / len(self.data_loader):.06f} \n")
            f.close()

            if self.current_epoch % 2 == 0 or self.current_epoch == self.max_epoch:
                self.save_all_models(epoch_num=self.current_epoch)

            self.current_epoch += 1

    def inference(self, epoch_num: int):
        """ Main function for testing.
        your trainer must implement `inference_sub` function.
        """

        assert self.is_training is False

        # prepare dataset.
        self.data = get_dataset(dataset_type=self.dataset_type, data_dir=self.data_dir, use_mask=self.use_mask)
        self.data_loader = DataLoader(
            self.data, shuffle=True, batch_size=self.batch_size, pin_memory=True, drop_last=True, num_workers=1,
        )
        self.batch_num = len(self.data_loader)

        # create output folders.
        save_path = self.out_dir_path.joinpath(f"images_ep{epoch_num:05d}", self.data.data_path.name)
        save_path.mkdir(parents=True)

        self.locate_all_models_to_device()
        self.load_all_models(epoch_num)
        self.set_mode_all_models(self.is_training)

        with torch.no_grad():
            self.inference_sub(save_path, self.batch_size)

    def inference_mesh(self, epoch_num: int, resolution: int, mesh_path: Path = None):
        """ This function export 3D mesh (.ply) from the trained SDF.

        Args:
            epoch_num (int): epoch number.
            resolution (int): how many positions you want to sample uniformly inside the bounding box.
                larger value generates more detailed mesh.
            mesh_path (Path): you can explicitly define the path of mesh.
        """

        assert self.is_training is False

        if mesh_path is None:
            save_path = self.out_dir_path.joinpath(f"epoch{epoch_num}_res{resolution}.ply")
        else:
            save_path = mesh_path

        self.locate_all_models_to_device()
        self.load_all_models(epoch_num)
        self.set_mode_all_models(self.is_training)

        with torch.no_grad():
            sdf = self.models_dic["sdfnet"].to(self.device)
            mesh = shapeutils.get_surface_trace(sdf=lambda x: sdf(x)[:, 0], resolution=resolution)
            mesh.export(str(save_path), "ply")

    def train_one_epoch(self, data_loader: DataLoader) -> float:
        """ This function defines your trainer's behavior in one epoch.
        All the trainer must be implemented with this function.

        Args:
            data_loader (DataLoader): data_loader.

        Returns:
            float: sum of the loss value in one epoch.
        """
        raise NotImplementedError

    def inference_sub(self, save_path: Path, split_size: int):
        """ This function defines what your trainer does in testing.
        please implement this function.
        """
        raise NotImplementedError

    def save_all_models(self, epoch_num: int):
        """ Save all the models in models_dic.
        """

        for model_name in self.models_dic:
            torch.save(
                self.models_dic[model_name].state_dict(),
                self.out_dir_path.joinpath(MODELS_FOLDER_NAME, f"ep{epoch_num:05d}_{model_name}.pth"),
            )

    def load_previous_training_models(self, path: Path, epoch_num: int, model_names: list):
        """ Load designated trained models of the previous stage.
        """
        for model_name in model_names:
            self.models_dic[model_name].load_state_dict(
                torch.load(
                    path.joinpath(MODELS_FOLDER_NAME, f"ep{epoch_num:05d}_{model_name}.pth"),
                    map_location=self.device)
            )

    def load_all_models(self, epoch_num: int):
        """ Load all the trained models according to the input epoch number.
        """

        for model_name in self.models_dic:
            self.models_dic[model_name].load_state_dict(
                torch.load(
                    self.out_dir_path.joinpath(MODELS_FOLDER_NAME, f"ep{epoch_num:05d}_{model_name}.pth"),
                    map_location=self.device,
                )
            )

    def locate_all_models_to_device(self):
        """ Locate all the models to the device.
        """

        for name in self.models_dic:
            self.models_dic[name].to(self.device)

    def set_mode_all_models(self, is_training: bool):
        """ Handle training/inferencing of the models.
        """

        for model_name in self.models_dic:
            self.models_dic[model_name].train(is_training)

            if not is_training:
                for param in self.models_dic[model_name].parameters():
                    param.requires_grad = False

    def set_all_optimizers(self):
        """ Set all optimizers to all the models.
        """
        for model_name in self.models_dic:
            self.optims_dic[model_name] = optim.Adam(self.models_dic[model_name].parameters(), lr=self.lr)

    def set_all_schedulers(self, decay_steps: int):
        """ set all schedulers to all the models.
        """
        for model_name in self.models_dic:
            self.scheduler_dic[model_name] = torch.optim.lr_scheduler.ExponentialLR(
                self.optims_dic[model_name], 0.1 ** (1. / decay_steps)
            )

    def update_all_models(self, loss: Tensor, clip_sdf_grad=False, skip_nan=False):
        """ Update all the models in self.models_dic.

        Args:
            loss (Tensor): computed loss.
            clip_sdf_grad (bool): if True, the gradient of SDF will be clipped.
            skip_nan (bool): if True, avoid updating the parameters when there is `nan` in the gradient.

        Note:
            in the third stage, clipping the gradient of SDF allows the training to converge.
        """

        for model_name in self.models_dic:
            for param in self.models_dic[model_name].parameters():
                param.grad = None

        loss.backward()

        if clip_sdf_grad:
            nn.utils.clip_grad_norm_(self.models_dic["sdfnet"].parameters(), 0.1)

        if skip_nan:
            for param in self.models_dic["sdfnet"].parameters():
                cur_grad = param.grad.data.mean()
                if torch.isnan(cur_grad):
                    print("skip the current batch")
                    for model_name in self.models_dic:
                        self.scheduler_dic[model_name].step()
                    return

        for model_name in self.optims_dic:
            self.optims_dic[model_name].step()
            self.scheduler_dic[model_name].step()

    def calc_grad_theta(self, rays_o: Tensor, rays_d: Tensor, z_samples_eik: Tensor) -> Tensor:
        """ This function computes gradient of SDF to compute eikonal loss.
        Refer to: https://github.com/lioryariv/volsdf/blob/main/code/model/network.py

        Args:
            rays_o (Tensor): the origin of the rays.
            rays_d (Tensor): the direction of the rays.
            z_samples_eik (Tensor): samples of 3D positions for computing Eikonal loss.

        Returns:
            Computed gradient.
        """

        n_eik_points = self.batch_size

        eikonal_points = torch.distributions.Uniform(
            -self.data.bounding_sphere_r, self.data.bounding_sphere_r).sample((n_eik_points, 3)).to(self.device)

        eik_near_points = (rays_o.unsqueeze(1) + z_samples_eik.unsqueeze(2) * rays_d.unsqueeze(1)).reshape(-1, 3)
        eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
        grad_theta = self.models_dic["sdfnet"].get_gradient(eikonal_points, self.data.bounding_sphere_r)

        return grad_theta

    def reshape_and_scale_all_images(self,
                                     valid_mask: Tensor or None,
                                     max_val: float or None,
                                     **kwargs: Tensor) -> dict:
        """ This function reshapes the outputs tensor and change the range of values from [0, 1] to [0, 65535].

        Args:
            valid_mask (Tensor or None): a mask pointing object areas. shapes can be (h*w, 1) or (h*w, 3).
            max_val (float or None): the images will be multiplied by this value and then clipped.
            kwargs (Tensor): the images you want to reshape and scale. shapes can be (h*w, 1) or (h*w, 3).

        Returns:
             (dict): a dictionary that includes every input image changed its shape and max values.
        """

        if valid_mask is not None:
            if valid_mask.shape[1] == 1:  # change the shape: (h*w, 1) --> (h*w, 3).
                valid_mask = valid_mask.repeat(1, 3)
            invalid_mask = torch.logical_not(valid_mask)

        else:
            invalid_mask = torch.zeros(self.data.height * self.data.width, 3, dtype=torch.bool)  # no invalid pixels.

        out_dic = {}
        for key in kwargs:
            img = kwargs[key]

            if img.shape[1] == 1:
                img = img.repeat(1, 3)  # change the shape: (h*w, 1) --> (h*w, 3).

            img[invalid_mask] = 0
            img = img.reshape(self.data.height, self.data.width, 3)

            if max_val is not None:
                img = torch.clip(img * max_val, 0, max_val)

            out_dic[key] = img

        return out_dic


class BlenderExportMixin:
    def export_animation_on_blender(self,
                                    epoch_num: int,
                                    resolution_mesh: int,
                                    angle: float,
                                    margin: float,
                                    uv_height: int,
                                    uv_width: int,
                                    focal_length: float,
                                    camera_path_duration: int,
                                    camera_path_radius: float,
                                    camera_path_z: float,
                                    camera_reso_x: int,
                                    camera_reso_y: int,
                                    env_name: str,
                                    rendering_mode: str,
                                    cycles_max_sampling: int) -> None:
        """ A function to render Blender animation.

        Args:
            epoch_num: epoch number.
            resolution_mesh: mesh resolution.
            angle: angle limit for smart UV projection in degrees.
            margin: margin to use between islands in the UV map.
            uv_height: height of uv texture.
            uv_width: width of uv texture.
            focal_length: focal length for the camera lens.
            camera_path_duration: number of frames it takes for the camera to complete one revolution around the path.
            camera_path_radius: radius of the circular path the camera will follow.
            camera_path_z: Z coordinate of the circular path's center.
            camera_reso_x: horizontal resolution of the camera.
            camera_reso_y: vertical resolution of the camera.
            env_name: environment map name.
            rendering_mode: rendering engine to use ("EEVEE" or "Cycles").
            cycles_max_sampling: maximum number of samples for Cycles rendering.

        Notes:
            after running this function, the following files will be saved based on the following structure:
                |- blender_epxxxxx /
                    |- env_name /
                        |- diffuse_albedo.png
                        |- roughness.png
                        |- default_mesh.ply
                        |- unwrapped_mesh.obj
                        |- unwrapped_mesh.mtl
                        |- albedo_roughness.mp4
                        |- roughness_vis.mp4
                        |- blender_scene.blend
        """

        out_base_path = self.out_dir_path.joinpath(
            f"blender_ep{epoch_num:05d}", Path(env_name).stem, f"mesh{resolution_mesh}_uv{uv_height}_{uv_width}"
        )
        out_base_path.mkdir(parents=True)

        default_mesh_name = "default_mesh.ply"
        unwrapped_mesh_name = "unwrapped_mesh.obj"
        albedo_name = "diffuse_albedo.png"
        roughness_name = "roughness.png"
        blender_scene_name = "blender_scene.blend"

        # generate mesh.
        self.inference_mesh(
            epoch_num=epoch_num, resolution=resolution_mesh, mesh_path=out_base_path.joinpath(default_mesh_name)
        )

        # unwrap the generated mesh.
        bl.uv_unwrapping_core(
            default_mesh_path=out_base_path.joinpath(default_mesh_name),
            unwrapped_mesh_path=out_base_path.joinpath(unwrapped_mesh_name),
            angle=angle,
            margin=margin,
        )

        # generate uv texture maps.
        self.inference_uv(
            epoch_num=epoch_num,
            mesh_path=out_base_path.joinpath(unwrapped_mesh_name),
            uv_height=uv_height,
            uv_width=uv_width,
            albedo_path=out_base_path.joinpath(albedo_name),
            roughness_path=out_base_path.joinpath(roughness_name),
            mtl_path=out_base_path.joinpath(f"{Path(unwrapped_mesh_name).stem}.mtl")
        )

        # load and apply textures.
        obj = bl.load_obj(obj_path=out_base_path.joinpath(unwrapped_mesh_name))
        bl.apply_texture_map(
            obj=obj,
            albedo_uv_path=out_base_path.joinpath(albedo_name),
            roughness_uv_path=out_base_path.joinpath(roughness_name)
        )
        bl.apply_vis_roughness_map(obj=obj, roughness_uv_path=out_base_path.joinpath(roughness_name))
        bl.modify_obj(obj)

        # set Blender scene.
        bl.set_circle_path_camera(
            focal_length=focal_length,
            camera_path_radius=camera_path_radius,
            camera_path_z=camera_path_z,
            camera_path_duration=camera_path_duration
        )
        bl.set_world(env_path=Path("env_maps").joinpath(env_name))

        # render.
        bl.set_rendering_mode(rendering_mode, True, cycles_max_sampling, camera_reso_x, camera_reso_y)
        bl.save_animation(obj, out_base_path)
        bl.save_blender_data(out_base_path.joinpath(blender_scene_name))
