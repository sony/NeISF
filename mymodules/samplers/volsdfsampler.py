# volsdfsampler.py
""" These classes define how to sample 3D positions from a ray, which introduced in volSDF [Yariv et al. 2021].
Refer to: https://github.com/lioryariv/volsdf/blob/main/code/model/ray_sampler.py.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from typing import Tuple

import torch
from torch import Tensor

from mymodules.models.volsdf import SDFNet
from .sampler_base import SamplerBase
from .utils import get_sphere_intersections


class UniformPosSampler(SamplerBase):
    """ this Sampler class uniformly samples 3D points between near and far bound.

    Attributes:
        is_foreground (bool): ``True`` for foreground sampler.

    Note:
        if is_foreground == True && use_background == True: far = sphere intersection.
        else: far = far.
    """
    def __init__(self,
                 near: float,
                 far: float,
                 n_samples: int,
                 bounding_sphere_r: float,
                 device: torch.device,
                 use_background: bool,
                 is_foreground: bool = True):

        if not is_foreground and not use_background:  # this sampler is for background, but use_background is false.
            raise ValueError("We do not accept both `is_foreground` and `use_background` are false.")

        if not use_background and far != bounding_sphere_r * 2.:
            raise ValueError("Currently far must be 2 * bounding_sphere_r")

        super(UniformPosSampler, self).__init__(
            near=near,
            far=far,
            n_samples=n_samples,
            device=device,
            bounding_sphere_r=bounding_sphere_r,
            use_background=use_background,
        )

        self.is_foreground = is_foreground

    def get_z_vals(self, rays_o: Tensor, rays_d: Tensor, is_training: bool) -> Tensor:
        """ generate the uniformly sampled 3D positions.

        Args:
            rays_o (Tensor): the origin of rays (b_size, 3).
            rays_d (Tensor): the directions of rays (b_size, 3).
            is_training (bool): in training, use `True`.

        Returns:
            (Tensor): uniformly sampled 3D positions (b_size, n_samples).
        """

        b_size = rays_o.shape[0]

        if self.is_foreground and self.use_background:  # calculate sphere intersection.
            far = get_sphere_intersections(rays_o=rays_o, rays_d=rays_d, r=self.bounding_sphere_r)  # (b_size, 1)

        elif self.is_foreground and not self.use_background:
            far = self.far * torch.ones(b_size, 1, device=self.device)  # (b_size, 1)

        else:  # for the background sampler, the samples must be [0, 0.002, ..., 1/R].
            # Refer to : https://arxiv.org/pdf/2010.07492.pdf, Fig 8.
            far = self.far * torch.ones(b_size, 1, device=self.device) / self.bounding_sphere_r  # (b_size, 1)

        # near bound is always the same.
        near = self.near * torch.ones(b_size, 1, device=self.device)  # (b_size, 1)

        t_vals = torch.linspace(0., 1., steps=self.n_samples, device=self.device)
        z_vals = near * (1. - t_vals) + far * t_vals  # (b_size, n_sample).

        if is_training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=self.device)

            z_vals = lower + (upper - lower) * t_rand

        return z_vals


class ErrorBoundPosSampler(SamplerBase):
    """ This Position Sampler samples positions based on VolSDF.

    For more details, refer to the following script:
    https://github.com/lioryariv/volsdf/blob/main/code/model/ray_sampler.py.

    Attributes:
        n_samples_eval (int):
        n_samples_extra (int):
        eps (float):
        beta_iter (int):
        max_iter (int):
        n_samples_inverse_sphere (int): the number of samples for the background.
        add_tiny (float):
    """

    def __init__(self,
                 near: float,
                 far: float,
                 n_samples: int,
                 n_samples_eval: int,
                 n_samples_extra: int,
                 device: torch.device,
                 bounding_sphere_r: float,
                 eps: float = 0.1,
                 beta_iter: int = 10,
                 max_iter: int = 5,
                 use_background: bool = False,
                 n_samples_inverse_sphere: int = 0,
                 add_tiny: float = 0.0):
        """ For some default values, please see the following part of the paper:
        https://arxiv.org/abs/2106.12052, sec. 3.4.
        """

        super(ErrorBoundPosSampler, self).__init__(
            near=near,
            far=far,
            n_samples=n_samples,
            device=device,
            bounding_sphere_r=bounding_sphere_r,
            use_background=use_background,
        )

        self.n_samples_eval = n_samples_eval
        self.n_samples_extra = n_samples_extra
        self.eps = eps
        self.beta_iter = beta_iter
        self.max_iter = max_iter
        self.n_samples_inverse_sphere = n_samples_inverse_sphere
        self.add_tiny = add_tiny

        self.uniform_pos_sampler = UniformPosSampler(
            near=near,
            far=far,
            n_samples=n_samples_eval,
            bounding_sphere_r=bounding_sphere_r,
            device=device,
            use_background=use_background,
        )

        if use_background:  # add another sampler for background.
            self.uniform_pos_sampler_bg = UniformPosSampler(
                near=0.,
                far=1.,
                bounding_sphere_r=bounding_sphere_r,
                n_samples=n_samples_inverse_sphere,
                device=device,
                use_background=use_background,
                is_foreground=False,
            )

    def get_z_vals(self,
                   rays_o: Tensor,
                   rays_d: Tensor,
                   sdf_net: SDFNet,
                   is_training: bool) -> Tuple[Tensor, Tensor]:
        """ Sample the 3D positions.

        Args:
            rays_o (Tensor): the origin of rays (b_size, 3).
            rays_d (Tensor): the directions of rays (b_size, 3).
            sdf_net (SDFNet):
            is_training (bool): if in the training process, True. otherwise False.

        Returns:
            (Tensor): sampled 3D positions for training.
            (Tensor): 3D positions for calculating Eikonal loss.
        """

        beta0 = sdf_net.density.get_beta().detach()

        # Initializing with uniform sampling (b_size, n_sample).
        z_vals = self.uniform_pos_sampler.get_z_vals(rays_d=rays_d, rays_o=rays_o, is_training=is_training)

        # Get maximum beta from the upper bound (Lemma 2).
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists ** 2.).sum(-1)
        beta = torch.sqrt(bound)

        # loop for updating beta. (Algorithm 1)
        iter_num = 0
        not_converge = True
        samples = z_vals  # (b_size, n_sample)
        samples_idx = None

        while not_converge and iter_num < self.max_iter:
            # (b_size * n_sample, 3)
            positions = (rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * samples.unsqueeze(2)).reshape(-1, 3)

            with torch.no_grad():
                samples_sdf_val = sdf_net.get_sdf_vals(positions, self.bounding_sphere_r)

            if samples_idx is None:
                sdf_val = samples_sdf_val
            else:
                sdf_merge = torch.cat([sdf_val.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                       samples_sdf_val.reshape(-1, samples.shape[1])], -1)
                sdf_val = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)

            # Calculating the bound d* (Theorem 1)
            d = sdf_val.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1, device=self.device)
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
            d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

            # Updating beta using line search
            curr_error = self.get_error_bound(beta0, sdf_net, sdf_val, z_vals, dists, d_star)
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iter):
                beta_mid = (beta_min + beta_max) / 2.
                curr_error = self.get_error_bound(beta_mid.unsqueeze(-1), sdf_net, sdf_val, z_vals, dists, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            density = sdf_net.density(sdf_val.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

            dists = torch.cat([dists, torch.tensor([1e10], device=self.device).unsqueeze(0).repeat(dists.shape[0], 1)],
                              -1)
            free_energy = dists * density
            shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1, device=self.device), free_energy[:, :-1]],
                                            dim=-1)
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = alpha * transmittance  # probability of the ray hits something here

            #  Check if we are done and this is the last sampling
            iter_num += 1
            not_converge = beta.max() > beta0

            if not_converge and iter_num < self.max_iter:
                # Sample more points proportional to the current error bound.
                ns = self.n_samples_eval
                bins = z_vals

                error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * (dists[:, :-1] ** 2.) / (
                        4 * beta.unsqueeze(-1) ** 2)
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * transmittance[:, :-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
            else:
                # Sample the final sample set to be sed in the volume rendering integral.
                ns = self.n_samples
                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

            # Invert CDF
            if (not_converge and iter_num < self.max_iter) or (not is_training):
                u = torch.linspace(0., 1., steps=ns, device=self.device).unsqueeze(0).repeat(cdf.shape[0], 1)
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [ns], device=self.device)
            u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, n_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = (cdf_g[..., 1] - cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

            # Adding samples if not converged
            if not_converge and iter_num < self.max_iter:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

        z_samples = samples
        near = self.near * torch.ones(rays_d.shape[0], 1, device=self.device)  # (b_size, 1)
        far = self.far * torch.ones(rays_d.shape[0], 1, device=self.device)  # (b_size, 1)

        if self.use_background:
            far = get_sphere_intersections(rays_o=rays_o, rays_d=rays_d, r=self.bounding_sphere_r)  # (b_size, 1)

        if self.n_samples_extra > 0:  # ??
            if is_training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.n_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1] - 1, self.n_samples_extra, device=self.device).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # add some near surface points
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],), device=self.device)
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

        if self.use_background:  # add background points.
            z_vals_bg = self.uniform_pos_sampler_bg.get_z_vals(rays_d=rays_d, rays_o=rays_o, is_training=is_training)
            z_vals = (z_vals, z_vals_bg)

        return z_vals, z_samples_eik

    def get_error_bound(self, beta, sdf_net, sdf_val, z_vals, dists, d_star):
        density = sdf_net.density(sdf_val.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1, device=self.device), dists * density[:, :-1]],
                                        dim=-1)
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(
            -integral_estimation[:, :-1])

        return bound_opacity.max(-1)[0]
