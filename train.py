# train.py
""" Run this script for training.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

USAGE:
    $ python train.py {YOUR CONFIG NAME}.json
"""

import argparse
from pathlib import Path

from mymodules.trainers import trainer_provider


parser = argparse.ArgumentParser()
parser.add_argument("config_name", type=str, help="the config name you want to use.")

if __name__ == "__main__":
    args = parser.parse_args()
    config_name = args.config_name

    config_path = Path("configs").joinpath(config_name)

    trainer = trainer_provider(config_path, is_train=True)
    trainer.train()
