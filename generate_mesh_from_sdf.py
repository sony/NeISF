# generate_mesh_from_sdf.py
""" This script extracts 3D mesh from trained SDF
based on the marching cube algorithm.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

USAGE:
    $ python generate_mesh_from_sdf.py {RESULT FOLDER NAME} {EPOCH NUM} {RESOLUTION}

NOTE:
    Resolution represents how many points will be sampled for the marching cube algorithm.
    The larger resolution generates more detailed mesh. In our experience, 800 can render decent 3D meshes.
"""

import argparse
from pathlib import Path

from mymodules.trainers import trainer_provider


parser = argparse.ArgumentParser()
parser.add_argument("result_folder", type=str, help="the folder name your models are stored.")
parser.add_argument("epoch_num", type=int)
parser.add_argument("resolution", type=int, help="larger number generates more detailed mesh. 800 is recommended.")


if __name__ == "__main__":
    args = parser.parse_args()
    result_folder = args.result_folder
    epoch_num = args.epoch_num
    resolution = args.resolution

    config_path = list(Path("results").joinpath(result_folder).glob("*_config.json"))[0]

    trainer = trainer_provider(config_path, is_train=False)
    trainer.inference_mesh(epoch_num=epoch_num, resolution=resolution)
