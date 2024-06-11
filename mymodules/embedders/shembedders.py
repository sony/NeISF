# shembedders.py
""" Classes for spherical-harmonics encoding.
Some lines may refer to: https://github.com/sxyu/svox2/blob/master/svox2/utils.py.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

import torch
from torch import Tensor

from .basicmbedders import Embedder


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


class SHEmbedder(Embedder):
    """ Class for computing spherical harmonics encoding.

    Attributes:
        sh_coords (tuple): the dimension of SH you want to use.
        out_dim (int): the number of encoded dimension.
    """

    def __init__(self, sh_coords: tuple):
        super(SHEmbedder, self).__init__()

        self.sh_coords = sh_coords
        self.out_dim = self.get_sh_dim(sh_coords)

    def embed(self, ref_d: Tensor, rho: Tensor) -> Tensor:
        """ returns the spherical harmonics encoding of the input vector.

        Args:
            ref_d (Tensor): the directions of reflection vector (n, 3).
            rho (Tensor): surface roughness (n, 1).

        Returns:
            (Tensor): the input direction embedded in the spherical harmonics encoding (n, sh_dim).

        Raises:
            ValueError when sh_coords includes numbers except 1, 2, or 4.
        """

        result = torch.empty((ref_d.shape[0], self.out_dim), dtype=ref_d.dtype, device=ref_d.device)

        x, y, z = ref_d[:, 0], ref_d[:, 1], ref_d[:, 2]
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z

        current_pos = 0

        for sh_coord in self.sh_coords:
            if sh_coord == 0:
                result[:, current_pos] = SH_C0
            elif sh_coord == 1:
                result[:, current_pos] = -SH_C1 * y
                result[:, current_pos + 1] = SH_C1 * z
                result[:, current_pos + 2] = -SH_C1 * x
            elif sh_coord == 2:
                ak = self.get_ak(l_num=1, rho=rho)
                result[..., current_pos] = SH_C2[0] * ak * xy
                result[..., current_pos + 1] = SH_C2[1] * ak * yz
                result[..., current_pos + 2] = SH_C2[2] * ak * (2.0 * zz - xx - yy)
                result[..., current_pos + 3] = SH_C2[3] * ak * xz
                result[..., current_pos + 4] = SH_C2[4] * ak * (xx - yy)
            elif sh_coord == 3:
                raise ValueError("sh_coord == 3 is currently not allowed!")
                # result[..., current_pos] = SH_C3[0] * y * (3 * xx - yy)
                # result[..., current_pos + 1] = SH_C3[1] * xy * z
                # result[..., current_pos + 2] = SH_C3[2] * y * (4 * zz - xx - yy)
                # result[..., current_pos + 3] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                # result[..., current_pos + 4] = SH_C3[4] * x * (4 * zz - xx - yy)
                # result[..., current_pos + 5] = SH_C3[5] * z * (xx - yy)
                # result[..., current_pos + 6] = SH_C3[6] * x * (xx - 3 * yy)
            elif sh_coord == 4:
                ak = self.get_ak(l_num=2, rho=rho)
                result[..., current_pos] = SH_C4[0] * ak * xy * (xx - yy)
                result[..., current_pos + 1] = SH_C4[1] * ak * yz * (3. * xx - yy)
                result[..., current_pos + 2] = SH_C4[2] * ak * xy * (7. * zz - 1.)
                result[..., current_pos + 3] = SH_C4[3] * ak * yz * (7. * zz - 3.)
                result[..., current_pos + 4] = SH_C4[4] * ak * (zz * (35. * zz - 30.) + 3.)
                result[..., current_pos + 5] = SH_C4[5] * ak * xz * (7. * zz - 3.)
                result[..., current_pos + 6] = SH_C4[6] * ak * (xx - yy) * (7. * zz - 1.)
                result[..., current_pos + 7] = SH_C4[7] * ak * xz * (xx - 3. * yy)
                result[..., current_pos + 8] = SH_C4[8] * ak * (xx * (xx - 3. * yy) - yy * (3. * xx - yy))
            else:
                raise ValueError("sh_coord includes wrong numbers!")

            current_pos += sh_coord * 2 + 1

        return result

    @staticmethod
    def get_sh_dim(sh_coords: tuple) -> int:
        """ Compute the number of dimension of SH.

        Args:
            sh_coords (tuple): which dimension you want to use.
        Returns:
            (Tensor) the number of dimension after the SH embedding.
        """

        sh_dim = 0
        for sh_coord in sh_coords:
            sh_dim += sh_coord * 2 + 1

        return sh_dim

    @staticmethod
    def get_ak(l_num: int, rho: Tensor) -> Tensor:
        """ Compute Al(k), described in Eq. 8 in ref-nerf.

        Refer to: https://arxiv.org/pdf/2112.03907.pdf.

        Args:
            l_num (int): l_th attenuation in SH encoding.
            rho (Tensor): roughness values (n, 1).
        Returns:
            (Tensor): A_k in the SH encoding (n,).
        """

        return torch.exp(-l_num * (l_num + 1) * rho / 2.).reshape(-1)
