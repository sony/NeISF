# evaluationutils.py
""" Library for evaluation.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import numpy as np


def calc_psnr(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None) -> float:
    """ Compute PSNR between img1 and img2.
    Note that images should be normalized into [0, 1].

    Args:
        img1 (np.ndarray): image (h, w, 3).
        img2 (np.ndarray): image (h, w, 3).
        mask (np.ndarray): image (h, w, 3). bool.

    Returns:
        (float): calculated PSNR.
    """

    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError("ndim must be 3.")
    if img1.shape[2] != 3 or img2.shape[2] != 3:
        raise ValueError("the third dimension must be 3.")
    if np.max(img1) > 1 or np.max(img2) > 1:
        raise ValueError("img must be normalized into [0, 1].")
    if np.min(img1) < 0 or np.min(img2) < 0:
        raise ValueError("img must be normalized into [0, 1].")

    if mask is not None:
        if mask.ndim != 3:
            raise ValueError("mask's ndim must be 3.")
        if mask.shape[2] != 3:
            raise ValueError("mask's third dimension must be 3.")
        if mask.dtype != bool:
            raise TypeError("mask should be bool array.")

    diff = img1 - img2  # (h, w, 3)

    if mask is not None:
        diff = diff[mask]  # (n,)

    return float(10 * np.log10(1 / np.mean(diff * diff)))


def calc_l1(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None) -> float:
    """ Compute mean absolute error between img1 and img2.

    Args:
        img1 (np.ndarray): image (h, w, 3).
        img2 (np.ndarray): image (h, w, 3).
        mask (np.ndarray): image (h, w, 3). bool.

    Returns:
        (float): calculated absolute error.
    """

    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError("ndim must be 3.")
    if img1.shape[2] != 3 or img2.shape[2] != 3:
        raise ValueError("the third dimension must be 3.")

    if mask is not None:
        if mask.ndim != 3:
            raise ValueError("mask's ndim must be 3.")
        if mask.shape[2] != 3:
            raise ValueError("mask's third dimension must be 3.")
        if mask.dtype != bool:
            raise TypeError("mask should be bool array.")

    diff = np.abs(img1 - img2)  # (h, w, 3)

    if mask is not None:
        diff = diff[mask]

    return float(diff.mean())


def calc_mae(normal1: np.ndarray, normal2: np.ndarray, mask: np.ndarray = None) -> float:
    """ Compute mean angular error (MAE) between normal1 and normal2.
    Note that images should be normalized into [-1, 1].

    Args:
        normal1 (np.ndarray): image (h, w, 3).
        normal2 (np.ndarray): image (h, w, 3).
        mask (np.ndarray): image (h, w, 3). bool.

    Returns:
        (float): calculated MEA [rad].
    """

    if normal1.ndim != 3 or normal2.ndim != 3:
        raise ValueError("ndim must be 3.")
    if normal1.shape[2] != 3 or normal2.shape[2] != 3:
        raise ValueError("the third dimension must be 3.")
    if np.max(normal1) > 1 or np.max(normal2) > 1:
        raise ValueError("img must be normalized into [-1, 1].")
    if np.min(normal1) < -1 or np.min(normal2) < -1:
        raise ValueError("img must be normalized into [-1, 1].")

    if mask is not None:
        if mask.ndim != 3:
            raise ValueError("mask's ndim must be 3.")
        if mask.shape[2] != 3:
            raise ValueError("mask's third dimension must be 3.")
        if mask.dtype != bool:
            raise TypeError("mask should be bool array.")

    dot = np.sum(normal1 * normal2, axis=2)  # (h, w)

    if mask is not None:
        dot = dot[mask[..., 0]]

    dot = np.clip(dot, -1, 1)

    theta = np.arccos(dot)

    return float(np.mean(theta))


def align_scale(img: np.ndarray, img_gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ Align the scale of the target image for scaling-invariant metric.
    Refer to: https://arxiv.org/pdf/2104.00674.pdf Eq.11

    Args:
        img (np.array): the estimated image of shape (h, w, 3).
        img_gt (np.array): the ground truth image of shape (h, w, 3).
        mask (np.array): object mask of shape (h, w, 3).

    Returns:
        (np.array): the scaled image of shape (h, w, 3).

    Notes:
        the range of  both the target and GT are assumed to be [0, 1].
    """

    scale_map = img_gt / (img + 1e-07)

    scale_r = np.median(scale_map[:, :, 0][mask[:, :, 0]])
    scale_g = np.median(scale_map[:, :, 1][mask[:, :, 1]])
    scale_b = np.median(scale_map[:, :, 2][mask[:, :, 2]])

    img = np.concatenate([img[:, :, 0:1] * scale_r, img[:, :, 1:2] * scale_g, img[:, :, 2:3] * scale_b], axis=-1)

    return np.clip(img, 0.0, 1.0)
