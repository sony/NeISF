# imageutils.py
""" Library for handling several types of images (I/O, range conversion, etc.).

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from pathlib import Path
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Note: this must be called before importing Opencv.

import numpy as np
import cv2
from matplotlib.pyplot import Figure


MAX_8BIT = 255.
MAX_16BIT = 65535.


def my_read_image(img_path: Path) -> np.ndarray:
    """ Return a loaded image according to the input path.

    Note that the color channels will be sorted as RGB order.
    And this function only accepts images with extensions `.png`, `.jpg`, or `.exr`.

    Args:
        img_path (Path): File path of the image you want to load.

    Returns:
        np.ndarray: Loaded image sorted as RGB order. Astype is numpy.float32.
    """
    if not isinstance(img_path, Path):
        raise TypeError("Input type must be Path object.")

    if img_path.suffix not in [".png", ".jpg", ".exr"]:
        raise TypeError("This function only accepts `.png`, `.jpg`, or `.exr`.")

    img = cv2.imread(str(img_path), -1)
    if img is None:
        raise FileNotFoundError(f"{img_path} not found.")

    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def my_write_image(img_path: Path, img: np.ndarray, max_val: float, min_val: float, astype: type) -> bool:
    """ Save the input ndarray as png format.

    Note that the color channels must be sorted as RGB order.

    Args:
        img_path (Path): File path of the image. Length must be shorter than 240.
        img (np.ndarray): ndarray of shape (h, w, 3). The channel must be in RGB order.
        max_val (float): the maximum value which your image must not go beyond.
        min_val (float): the minimum value which your image must not go beyond.
        astype (type): astype of the image.

    Returns:
        bool: True if the image is successfully saved.
    """

    if not isinstance(img_path, Path):
        raise TypeError("img_path must be Path object.")
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be np.ndarray.")
    if img.dtype != 'float32':
        raise TypeError("img.dtype must be float32.")
    if len(str(img_path)) > 240:
        raise ValueError("file path must be smaller than 240")
    if not img_path.parent.exists():
        raise FileNotFoundError("we cannot find your folder.")
    if img_path.exists():
        raise FileExistsError(f"{img_path} already exists.")
    if img.ndim != 3:
        raise ValueError("the image's dimension must be 3.")
    if img.shape[2] != 3:
        raise ValueError("the image must have RGB channels.")
    if np.min(img) < min_val or np.max(img) > max_val:
        raise ValueError("Your input array's range doesn't match to 16bit.")

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(str(img_path), img.astype(astype))


def my_write_image_16bit(img_path: Path, img: np.ndarray) -> bool:
    """ Save the input ndarray as 16bit-png format.

    Note that the color channels must be sorted as RGB order.

    Args:
        img_path (Path): File path of the image. Length must be shorter than 240.
        img (np.ndarray): ndarray of shape (h, w, 3). The channel must be in RGB order.

    Returns:
        bool: True if the image is successfully saved.
    """

    return my_write_image(img_path, img, max_val=MAX_16BIT, min_val=0., astype=np.uint16)


def my_write_image_8bit(img_path: Path, img: np.ndarray) -> bool:
    """ Save the input ndarray as 8bit-png format.

    Note that the color channels must be sorted as RGB order.

    Args:
        img_path (Path): File path of the image. Length must be shorter than 240.
        img (np.ndarray): ndarray of shape (h, w, 3). The channel must be in RGB order.

    Returns:
        bool: True if the image is successfully saved.
    """

    return my_write_image(img_path, img, max_val=MAX_8BIT, min_val=0., astype=np.uint8)


def my_write_image_exr(img_path: Path, img: np.ndarray) -> bool:
    """ Save the input ndarray as exr format.

    Note that the color channels must be sorted as RGB order.

    Args:
        img_path (Path): File path of the image. Length must be shorter than 240.
        img (np.ndarray): ndarray of shape (h, w, 3). The channel must be in RGB order.

    Returns:
        bool: True if the image is successfully saved.
    """
    return my_write_image(img_path, img, max_val=float("inf"), min_val=-float("inf"), astype=np.float32)


def write_all_images_in_dict(save_path: Path, idx: int, images_dict: dict) -> bool:
    """ Save all the images in the input dict.

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
        image_is_saved = my_write_image_16bit(
            save_path.joinpath(f"{idx + 1:03d}_{image_name}.png"),
            images_dict[image_name].detach().numpy(),
        )
        all_image_is_saved *= image_is_saved

    return all_image_is_saved


def my_write_plt_fig(plt_fig: Figure, img_path: Path):
    """ Save matplotlib figure in png format.

    Args:
        plt_fig (Figure): matplotlib figure.
        img_path (Path): File path you want to save the image.
    """

    if img_path.exists():
        raise FileExistsError(f"{img_path} already exists.")

    plt_fig.savefig(img_path, dpi=300, bbox_inches="tight")


def rgb_to_srgb(img: np.ndarray) -> np.ndarray:
    """ Convert the image's color space from liner-RGB to sRGB.

    Args:
          img (np.ndarray): input image. color space must be linear-RGB.

    Returns:
        np.ndarray: an image converted to sRGB color space.
    """

    if np.max(img) > 1:
        raise ValueError("Input image must be normalized into (0, 1).")

    a = 0.055

    high_mask = (img > 0.0031308)

    low_c = 12.92 * img
    high_c = (1 + a) * np.power(img, 1.0 / 2.4) - a

    low_c[high_mask] = high_c[high_mask]

    return low_c
