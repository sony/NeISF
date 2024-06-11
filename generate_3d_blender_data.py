# generate_3d_blender_data.py
""" This script generates the followings from a trained models:
    - 3D mesh.
    - uv texture maps.
    - animation rendered by Blender.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php


USAGE:
    $ python generate_3d_blender_data.py {RESULT FOLDER NAME} {EPOCH NUM} {RESOLUTION} {ENV NAME}

    for the other optional args, please see 'help'.

NOTE:
    after running this function, the following files will be saved based on the following structure:
        |- blender_epxxxxx /
            |- env_name /
                |- diffuse_albedo.png  # uv map of diffuse albedo.
                |- roughness.png  # uv map of roughness.
                |- default_mesh.ply  # exported mesh without uv unwrapping.
                |- unwrapped_mesh.obj  # mesh with uv unwrapping.
                |- unwrapped_mesh.mtl  # .mtl file for unwrapped_mesh.obj
                |- albedo_roughness.mp4  # rendered movie.
                |- roughness_vis.mp4  # rendered movie with only roughness.
                |- blender_scene.blend  # Blender scene file.
"""

import argparse

from mymodules.trainers import trainer_provider
from mymodules.trainers.trainers_base import SAVED_CONFIG_NAME, RESULT_PARENT_PATH

parser = argparse.ArgumentParser()
parser.add_argument("result_folder", type=str, help="the folder name your models are stored.")
parser.add_argument("epoch_num", type=int)
parser.add_argument("resolution_mesh", type=int, help="larger number generates more detailed mesh. 800 is recommended.")
parser.add_argument("env_name", type=str, help="the name of environment map. must be in env_maps folder.")

parser.add_argument("--angle", type=int, default=66, help="angle limit in Smart UV Project")
parser.add_argument("--margin", type=float, default=0.008, help="island margin in Smart UV Project")

parser.add_argument("--uv_height", type=int, default=4096, help="y resolution of UV map")
parser.add_argument("--uv_width", type=int, default=4096, help="x resolution of UV map")

parser.add_argument("--focal_length", type=float, default=150, help="focal length of camera.")
parser.add_argument("--camera_path_duration", type=int, default=200,
                    help="the time it takes for the camera to go around the circle path.")
parser.add_argument("--camera_path_radius", type=float, default=8, help="radius of camera circle path.")
parser.add_argument("--camera_path_z", type=float, default=4, help="z position of camera circle path.")
parser.add_argument("--camera_reso_x", type=int, default=1920, help="camera resolution of x. default 1920.")
parser.add_argument("--camera_reso_y", type=int, default=1080, help="camera resolution of y. default 1080.")
parser.add_argument("--rendering_mode", type=str, default="EEVEE",
                    help="rendering mode in blender. Either EEVEE or Cycles can be chosen. default EEVEE.")
parser.add_argument("--use_gpu", type=bool, default=True, help="whether to use GPU in Cycles rendering mode.")
parser.add_argument("--cycles_max_sampling", type=int, default=512,
                    help="number of max sampling in Cycles rendering mode.")

if __name__ == "__main__":
    args = parser.parse_args()

    config_path = RESULT_PARENT_PATH.joinpath(args.result_folder, SAVED_CONFIG_NAME)
    trainer = trainer_provider(config_path, is_train=False)

    trainer.export_animation_on_blender(
        epoch_num=args.epoch_num,
        resolution_mesh=args.resolution_mesh,
        angle=args.angle,
        margin=args.margin,
        uv_height=args.uv_height,
        uv_width=args.uv_width,
        focal_length=args.focal_length,
        camera_path_duration=args.camera_path_duration,
        camera_path_radius=args.camera_path_radius,
        camera_path_z=args.camera_path_z,
        camera_reso_x=args.camera_reso_x,
        camera_reso_y=args.camera_reso_y,
        env_name=args.env_name,
        rendering_mode=args.rendering_mode,
        cycles_max_sampling=args.cycles_max_sampling,
    )
