# NeISF: Neural Incident Stokes Field for Geometry and Material Estimation <!-- omit in toc -->
[Chenhao Li](<mailto:lch@is.ids.osaka-u.ac.jp>)<sup>1,3</sup>, 
[Taishi Ono](<mailto:taishi.ono@sony.com>)<sup>2</sup>, 
Takeshi Uemori<sup>1</sup>, 
Hajime Mihara<sup>1</sup>, 
Alexander Gatto<sup>2</sup>,
Hajime Nagahara<sup>3</sup>,
and Yusuke Moriuchi<sup>1</sup>

<sup>1</sup> Sony Semiconductor Solutions Corporation, 
<sup>2</sup> Sony Europe B.V., 
<sup>3</sup> Osaka University

This project is the implementation of "NeISF: Neural Incident Stokes Field for Geometry and Material Estimation (CVPR 2024)", which is a novel multi-view inverse rendering framework that reduces ambiguities using polarization cues.

### [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Li_NeISF_Neural_Incident_Stokes_Field_for_Geometry_and_Material_Estimation_CVPR_2024_paper.html) | [Data](https://sonyjpn.sharepoint.com/sites/S110-NeISF) <!-- omit in toc -->


## Table of Contents <!-- omit in toc -->
- [Dependencies](#dependencies)
- [Folder structure](#folder-structure)
- [Preparation](#preparation)
- [Run](#run)
  - [Training](#training)
  - [Testing](#testing)
  - [Evaluation](#evaluation)
  - [Exporting 3D mesh from trained SDFs](#exporting-3d-mesh-from-trained-sdfs)
  - [Re-lighting](#re-lighting)
  - [Exporting UV textures and Blender rendered animation](#exporting-uv-textures-and-blender-rendered-animation)
- [Dataset](#dataset)
  - [Polarized images](#polarized-images)
  - [Masks](#masks)
  - [Camera poses](#camera-poses)
  - [Camera normalization](#camera-normalization)
- [Config files](#config-files)
  - [Common parameters](#common-parameters)
  - [TrainerNeISF](#trainerneisf)
- [License](#license)
- [Citation](#citation)
<!--- [Implement a new trainer](#implement-a-new-trainer)-->


## Dependencies 
- Python 3.10 or newer (tested on 3.10.4 and 3.10.8).
- CUDA 10.1 or newer (tested on 11.3 and 10.1).

For the other dependencies, please see [requirements.txt](./requirements.txt).

## Folder structure
Our scripts assume the following folder structure and file names. 
See also [images/sample_folder](./images/sample_folder).
```
|- train.py
|- inference.py
|- ...
|- mymodules/
|- configs/
|- configs_sample/
|- results/
|- images/
    |- folder_1/
        |- poses_bounds.npy
        |- images_s0/
            |- img_001.exr  # please follow this naming convention.
            |- img_002.exr
            |- ...
        |- images_s1/
            |- img_001.exr
            |- img_002.exr
            |- ...
        |- images_s2/
            |- img_001.exr
            |- img_002.exr
            |- ...
        |- masks/
            |- img_001.png  # 16-bit, 3 channels.
            |- img_002.png
            |- ...
```

## Preparation
Install all the dependency using [requirements.txt](./requirements.txt), 
or, if you are a Docker user, you can use [Dockerfile](./Dockerfile).

Copy all the config files from `configs_sample/` to `configs/`.

## Run
### Training
This project includes three trainers: `TrainverVolSDF`, `TrainerNeISF`, and `TrainerNeISFNoStokes`.

For example, if you want to use `TrainverVolSDF`:
1. Edit `configs/trainervolsdf_config.json`.
2. Run the following command:
   ```
   $ python train.py trainervolsdf_config.json
   ```

As described in the paper (Sec. 4.6), the full pipeline of NeISF is composed of the following three steps:

1. train `TrainverVolSDF` using `trainervolsdf_config.json`.
2. train `TrainerNeISF` using `trainer_neisf_init_config.json`.
3. train `TrainerNeISF` using `trainer_neisf_joint_config.json`.

To reproduce our results, run these three trainings with using the written parameters.

If you want to train `TrainerNeISFNoStokes`, please do the step 2 and 3 with using `trainer_neisfnostokes_init_config.json`
and `trainer_neisfnostokes_joint_config.json`, respectively.


### Testing

```
$ python inference.py {RESULT FOLDER NAME} {IMAGE FOLDER NAME} {EPOCH NUM} -b {BATCH SIZE}
```
### Evaluation

```
$ python compute_metrics.py {RESULT FOLDER NAME} {IMAGE FOLDER NAME} {EPOCH NUM} -l {IMAGE_DOMAIN1} {IMAGE_DOMAIN2} ...
```
About the args or metrics, more details can be found in [compute_metrics.py](./compute_metrics.py).

### Exporting 3D mesh from trained SDFs
Run the following command:
 ```
 $ python generate_mesh_from_sdf.py {RESULT FOLDER NAME} {EPOCH_NUM} {resolution}
 ```

 ### Re-lighting
 1. Locate your environment illumination map (must be EXR format) under [env_maps](./env_maps/).
 2. Run the following command:

 ```
 $ python generate_relighting_image.py {YOUR RESULT FOLDER} {TARGET FOLDER} {EPOCH NUM} {ENV MAP NAME} -b {B SIZE} -l {SAMPLE ILLUM NUM}
 ```

 ### Exporting UV textures and Blender rendered animation
 1. Locate your environment illumination map (must be EXR format) under [env_maps](./env_maps/).
 2. Run the following command:
```
$ python generate_3d_blender_data.py {YOUR RESULT FOLDER} {EPOCH NUM} {MESH RESOLUTION} {ENV MAP NAME}
```

Known issues: We have observed some environments where this script can not correctly render videos. In this case, you may use Dockerfile.

## Dataset
Here we describe how to prepare your own dataset.

### Polarized images
Please see our appendix about how we created our HDR polarized dataset. In the same way as the convention, s0, s1, and s2 images representing the Stokes vectors are defined as follows:

   - s0 = (i_000 + i_045 + i_090 + i_135) / 2.
   - s1 = i_000 - i_090
   - s2 = i_045 - i_135

Save the images according to the [Folder structure](#folder-structure). Please also check [sample_folder](./images/sample_folder).


### Masks
Our method requires binary masks as inputs, the masks must follow the following rules:

- 16bit png with three channels.
- maximum intensity (white) represents valid pixels, minimum intensity (black) represents invalid ones.
- The same resolution as the polarized images.

Please also check [sample_folder](./images/sample_folder).


### Camera poses
We use the same format as [LLFF](https://github.com/Fyusion/LLFF) to describe the camera extrinsic and intrinsic parameters.
`poses_bounds.npy` stores a numpy array of size Nx17 (where N is the number of input images).

Assuming we have the following world-to-camera matrix (R), camera position (t), and camera intrinsics (h, w, f):
```
R = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
t = [t0, t1, t2]
height = h, width = w, focal length = f
```
then, one of the rows in `poses_bounds.npy` becomes like:
```
poses_bounds[i, :] = [r00, r01, r02, t0, h, r10, r11, r12, t1, w, r20, r21, r22, t2, f, 0, 0]
```
- The last two elements, which are used in LLFF to compute the near and far bound, are not used 
in this project.
- `[x,y,z]` axes of the camera point `[down, right, backwards]`.

### Camera normalization
Our scripts assume the following conditions:

- All the camera positions are located inside the sphere of radius=3.
- All the cameras are not looking inside out.

To assure the first assumption, please normalize your cameras by using the following command:

```
$ python preprocess_camera_normalization.py --flist {YOUR_DATA_DIR1} {YOUR_DATA_DIR2} {YOUR_DATA_DIR3} ...
```

This script will compute the viewing point by using the z-axes of all the cameras, shift the point to the origin, and then normalize all the cameras.
If you want to normalize several directories at the same time, for instance you have a training scene and an evaluation scene, please input multiple directories as in the example above.

For visualizing your cameras, use the following command:

```
$ python visualize_cameras.py {YOUR_DATA_DIR} {DATASET_TYPE}
```

Currently, only `neisf` is allowed for `DATASET_TYPE`.

## Config files
Here we describe the parameters included in the config files.

### Common parameters
| parameter name            | definition                                                                |
|---------------------------|---------------------------------------------------------------------------|
| trainer_name              | the name of the trainer                                                   |
| data_dir                  | the name of the directory for training                                    |
| dataset_type              | the type of the dataset. the current implementation only accepts `neisf`. |
| experiment_name           | the name of your result folder.                                           |
| batch_size                | the number of sampled pixels in one iteration.                            |
| max_epoch                 | the maximum number of epoch.                                              |
| sample_num                | the number of 3D points sampled along one ray.                            |
| positional_encoding_x_res | the dimension of positional encoding fot the 3D position x.               |
| positional_encoding_d_res | the dimension of positional encoding fot ray direction d.                 |
| gpu_num                   | the number of GPUs (multi-GPU is not supported).                          |
| lr                        | learning rate.                                                            |
| weights                   | a dictionary to store all the weight values. `eik_weight` is necessary.   |
| use_mask                  | if true, invalid pixels are not sampled (false is not well tested).       |

### TrainerNeISF
| parameter name           | definition                                            |
|--------------------------|-------------------------------------------------------|
| previous_stage_dir       | experiment name of the previous stage                 |
| previous_stage_epoch_num | which epoch should be loaded                          |
| stage_name               | name of the current training stage: `init` or `joint` |
| max_step_ray_march       | the number of ray-marching steps                      |


## License
This software is released under the MIT License. See [LICENSE](./LICENSE) for details.

## Citation
```
@InProceedings{Li_NeISF_CVPR2024,
    author    = {Li, Chenhao and Ono, Taishi and Uemori, Takeshi and Mihara, Hajime and Gatto, Alexander and Nagahara, Hajime and Moriuchi, Yusuke},
    title     = {NeISF: Neural Incident Stokes Field for Geometry and Material Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {21434-21445}
}
```
