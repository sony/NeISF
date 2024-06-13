# generate_relighting_image.py
""" This script renders the scene with an arbitrary environmen map.

USAGE:
    $ python generate_relighting_image.py {RESULT FOLDER NAME} {IMAGE FOLDER NAME} {EPOCH NUM} {ENV MAP NAME} \
        -b {BATCH SIZE} -l {ILLUMINATION SAMPLE NUM}

NOTE:
    to get high-quality images, it is recommended to use `--light_num` larger than 10000.
    if you encounter GPU memory issue, please use smaller `--batch_size` instead of reducing `--light_num`.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import argparse

from mymodules.trainers import trainer_provider
from mymodules.trainers.trainers_base import SAVED_CONFIG_NAME, RESULT_PARENT_PATH


parser = argparse.ArgumentParser()
parser.add_argument("result_folder", type=str, help="the folder name your models are stored.")
parser.add_argument("image_folder", type=str, help="the folder name your images are stored.")
parser.add_argument("epoch_num", type=int)
parser.add_argument("env_map_name", type=str, help="the name of environment map.")
parser.add_argument("-b", "--batch_size", type=int, default=1000)
parser.add_argument("-l", "--light_num", type=int, default=10000)


if __name__ == "__main__":
    args = parser.parse_args()
    result_folder = args.result_folder
    image_folder = args.image_folder
    epoch_num = args.epoch_num
    env_map_name = args.env_map_name
    batch_size = args.batch_size
    light_num = args.light_num

    config_path = RESULT_PARENT_PATH.joinpath(result_folder, SAVED_CONFIG_NAME)

    trainer = trainer_provider(config_path, is_train=False, inference_folder=image_folder, inference_batch=batch_size)
    trainer.render_relighting(image_folder, epoch_num, env_map_name, light_num)
