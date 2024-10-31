# compute_metrics.py
""" This script calculates & saves evaluation metrics for NeISF results.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php

USAGE:
    $ python compute_metrics.py {TARGET_DIR} {SRC_DIR} {EPOCH_NUM} -l {IMAGE_NAME1} {IMAGE_NAME2} ...

    for example: \n
    $ python compute_metrics.py bunny_neisf_joint bunny_eval 2000 -l rgb diffuse specular albedo roughness normal s0 s1 s2

Note:
    the args "IMAGE_NAME" must be one of them:
        {"s0", "s1", "s2", "albedo", "roughness", "diffuse", "specular", "rgb", "normal"}

    For each image domain, we use the following metrics:
        - sRGB, diffuse, and specular: PSNR.
        - albedo and roughness: a scale-invariant L1.
        - Stokes vectors: L1.
        - surface normal: mean angular error (MAE).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mymodules.tensorutils import gamma
import mymodules.evaluationutils as evalutil
from mymodules.imageutils import MAX_16BIT, my_read_image


parser = argparse.ArgumentParser()
parser.add_argument("result_folder", type=str, help="the directory including the estimated images.")
parser.add_argument("src_folder", type=str, help="the directory including the ground truths.")
parser.add_argument("epoch_num", type=int)
parser.add_argument("-m", "--use_mask", type=bool, default=True)
parser.add_argument("-l", "--domain_names", nargs="+", help="the name list of images", required=True)

DOMAIN_NAMES_MUST_BE = [
    "s0", "s1", "s2", "albedo", "roughness", "diffuse", "specular", "rgb", "normal"
]

def gamma_correct(img):
    img_tensor = torch.from_numpy(img)
    img_gamma = gamma(img_tensor).numpy()
    return img_gamma

if __name__ == '__main__':
    args = parser.parse_args()
    result_folder = args.result_folder
    src_folder = args.src_folder
    epoch_num = args.epoch_num
    use_mask = args.use_mask
    domain_names = args.domain_names

    est_path = Path("results").joinpath(result_folder, "images_ep{:05d}".format(epoch_num), src_folder)
    src_path = Path("images").joinpath(src_folder)

    # check
    if not est_path.exists():
        raise FileNotFoundError("target folder does not exist.")
    if not src_path.exists():
        raise FileNotFoundError("src file does not exist.")
    if est_path.joinpath("metric.json").exists():
        raise FileExistsError("metric.json already exists.")

    for domain_name in domain_names:
        if domain_name not in DOMAIN_NAMES_MUST_BE:
            raise ValueError(f"{domain_name} is not in {DOMAIN_NAMES_MUST_BE}.")

    # initialize result dictionary for saving the errors.
    result_dict = {f"Average Error for {domain_name}": 0 for domain_name in domain_names}

    # main loop.
    num_images = len(list(src_path.joinpath("images").glob("*.png")))

    for ii in range(num_images):
        if use_mask:
            mask_path = src_path.joinpath("masks").joinpath(f"img_{ii+1:03d}.png")
            mask_img = my_read_image(mask_path) / MAX_16BIT
            mask_img = (mask_img > 0.999)
        else:
            mask_img = None

        image_err_dict = {}  # a sub dict for saving each image's errors.

        for domain_name in domain_names:
            if domain_name in ["s0", "s1", "s2"]:
                src_img_path = src_path.joinpath(f"images_{domain_name}", f"img_{ii+1:03d}.exr")
                est_img_path = est_path.joinpath(f"{ii+1:03d}_{domain_name}.exr")
                src_img = my_read_image(src_img_path)
                est_img = my_read_image(est_img_path)

                err = evalutil.calc_l1(est_img, src_img, mask_img)
                image_err_dict[est_img_path.stem] = f"L1 error: {err:.5f}"

            elif domain_name in ["diffuse", "specular"]:
                src_img_path = src_path.joinpath(f"{domain_name}s", f"img_{ii+1:03d}.png")
                est_img_path = est_path.joinpath("{:03d}_{}.png".format(ii+1, domain_name))
                src_img = my_read_image(src_img_path) / MAX_16BIT
                est_img = my_read_image(est_img_path) / MAX_16BIT
                est_img = gamma_correct(est_img)

                err = evalutil.calc_psnr(est_img, src_img, mask_img)
                image_err_dict[est_img_path.stem] = f"PSNR: {err:.5f}"

            elif domain_name in ["albedo", "roughness"]:
                src_img_path = src_path.joinpath(f"{domain_name}s", f"img_{ii+1:03d}.png")
                est_img_path = est_path.joinpath("{:03d}_{}.png".format(ii+1, domain_name))
                src_img = my_read_image(src_img_path) / MAX_16BIT
                est_img = my_read_image(est_img_path) / MAX_16BIT

                # align scale.
                est_img_scaled = evalutil.align_scale(img=est_img, img_gt=src_img, mask=mask_img)

                err = evalutil.calc_l1(est_img_scaled, src_img, mask_img)
                image_err_dict[est_img_path.stem] = f"Scaled L1 error: {err:.5f}"

            elif domain_name == "rgb":
                src_img_path = src_path.joinpath("images", f"img_{ii+1:03d}.png")
                est_img_path = est_path.joinpath("{:03d}_{}.png".format(ii+1, domain_name))
                src_img = my_read_image(src_img_path) / MAX_16BIT
                est_img = my_read_image(est_img_path) / MAX_16BIT

                err = evalutil.calc_psnr(est_img, src_img, mask_img)
                image_err_dict[est_img_path.stem] = f"PSNR: {err:.5f}"

            elif domain_name == "normal":
                src_img_path = src_path.joinpath(f"{domain_name}s", f"img_{ii+1:03d}.png")
                est_img_path = est_path.joinpath("{:03d}_{}.png".format(ii+1, domain_name))

                src_img = (my_read_image(src_img_path) / MAX_16BIT) * 2. - 1.
                est_img = (my_read_image(est_img_path) / MAX_16BIT) * 2. - 1.

                src_img /= np.linalg.norm(src_img, axis=2, keepdims=True)
                est_img /= np.linalg.norm(est_img, axis=2, keepdims=True)

                err = evalutil.calc_mae(est_img, src_img, mask_img)
                err = np.rad2deg(err)
                image_err_dict[est_img_path.stem] = f"MAE [deg]: {err:.5f}"

            else:
                raise ValueError(f"Unacceptable existing domain name: {domain_name}")

            result_dict[f"Average Error for {domain_name}"] += err / num_images

        result_dict[f"error dict img {ii+1:03d}"] = image_err_dict

    with open(est_path.joinpath("metric.json"), "w") as outfile:
        json.dump(result_dict, outfile, indent=4)
