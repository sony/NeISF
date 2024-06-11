# inference.py
""" Run this script for inferencing.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

USAGE:
    for TrainerVolSDF, use the following:
    $ python inference.py {RESULT FOLDER NAME} {IMAGE FOLDER NAME} {EPOCH NUM} -b {BATCH SIZE}

    for TrainerNeISF or TrainerNeISFNoStokes, use the following:
    $ python inference.py {RESULT FOLDER NAME} {IMAGE FOLDER NAME} {EPOCH NUM} -b {BATCH SIZE} --illum_sample_num {ILLUM SAMPLE NUM}

NOTE:
    - The batch size is optional. Please change it according to your GPU. The default value is 1000.
    - Since we uniformly sample illumination, the number of illumination sample heavily affects the rendering quality.
    Please use as large number as possible.
"""

import argparse

from mymodules.globals import RESULT_PARENT_PATH
from mymodules.trainers import trainer_provider, SAVED_CONFIG_NAME


parser = argparse.ArgumentParser()
parser.add_argument("result_folder", type=str, help="the name of your result folder.")
parser.add_argument("image_folder", type=str, help="the name of the image folder.")
parser.add_argument("epoch_num", type=int)
parser.add_argument("-b", "--batch_size", type=int, default=1000)
parser.add_argument(
    "--illum_sample_num", type=int, default=None,
    help="the number of sample for illumination. only for TrainerNeISF or TrainerNeISFNoStokes"
)


if __name__ == "__main__":
    args = parser.parse_args()
    result_folder = args.result_folder
    image_folder = args.image_folder
    epoch_num = args.epoch_num
    batch_size = args.batch_size
    illum_sample_num = args.illum_sample_num

    config_path = RESULT_PARENT_PATH.joinpath(result_folder, SAVED_CONFIG_NAME)

    trainer = trainer_provider(
        config_path,
        is_train=False,
        inference_folder=image_folder,
        inference_batch=batch_size,
        inference_illum_sample=illum_sample_num,
    )
    trainer.inference(epoch_num=epoch_num)
