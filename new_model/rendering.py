from typing import Optional, Sequence

import torch
import nerfacc
from nerfacc.estimators.occ_grid import OccGridEstimator

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from torch import Tensor

from .cuda import is_cub_available
from .pack import pack_info
from .scan import exclusive_prod, exclusive_sum


def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    all_rays_o: torch.Tensor,
    all_rayd_d: torch.Tensor,
    train_chunk_size: int,
    # rendering options
    near_plane: float,
    far_plane: float,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""


    results = []
    chunk = (
        train_chunk_size
        if radiance_field.training
        else test_chunk_size
    )
    # run netwrok chunk by chunk to avoid OOM
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)

        rays_o = all_rays_o[i : i + chunk]
        rays_d = all_rayd_d[i : i + chunk]
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

            sigmas = radiance_field.query_density(positions)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

            rgbs, sigmas = radiance_field(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )

        # print("rays inds", ray_indices.shape, ray_indices[:7])
        # print("t_starts", t_starts.shape, t_starts[:7])
        # print("t_ends", t_ends.shape, t_ends[:7])
        n_rays = rays_o.shape[0]
        rgb, opacity, depth, extras = upd_rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=n_rays,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
            expected_depths=False
        )

        # t_k, center of the segments
        z_vals = (t_starts + t_ends)[:, None] / 2.0
        depth_std_sq = nerfacc.volrend.accumulate_along_rays(
            weights, (z_vals - depth[ray_indices]).square(),
            ray_indices=ray_indices, n_rays=n_rays
        )

        # print(rgb.shape, opacity.shape, depth.shape)
        # this just a cnter of the time segments
        # z_vals =
        chunk_results = [rgb, opacity, depth, depth_std_sq, len(t_starts)]
        results.append(chunk_results)


    colors, opacities, depths, depth_std_sq, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        depth_std_sq.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )

# this functions is copied from current version of nerfacc
# now it supports accumalated and expected depth
def upd_rendering(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    rgb_alpha_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[Tensor] = None,
    expected_depths: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`.

    This function is differentiable to the outputs of `rgb_sigma_fn` so it can
    be used for gradient-based optimization. It supports both batched and flattened input tensor.
    For flattened input tensor, both `ray_indices` and `n_rays` should be provided.


    Note:
        Either `rgb_sigma_fn` or `rgb_alpha_fn` should be provided.

    Warning:
        This function is not differentiable to `t_starts`, `t_ends` and `ray_indices`.

    Args:
        t_starts: Per-sample start distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        t_ends: Per-sample end distance. Tensor with shape (n_rays, n_samples) or (all_samples,).
        ray_indices: Ray indices of the flattened samples. LongTensor with shape (all_samples).
        n_rays: Number of rays. Only useful when `ray_indices` is provided.
        rgb_sigma_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and density
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        rgb_alpha_fn: A function that takes in samples {t_starts, t_ends,
            ray indices} and returns the post-activation rgb (..., 3) and opacity
            values (...,). The shape `...` is the same as the shape of `t_starts`.
        render_bkgd: Background color. Tensor with shape (3,).
        expected_depths: If True, return the expected depths. Else, the accumulated depth is returned.

    Returns:
        Ray colors (n_rays, 3), opacities (n_rays, 1), depths (n_rays, 1) and a dict
        containing extra intermediate results (e.g., "weights", "trans", "alphas")

    Examples:

    .. code-block:: python

        >>> t_starts = torch.tensor([0.1, 0.2, 0.1, 0.2, 0.3], device="cuda:0")
        >>> t_ends = torch.tensor([0.2, 0.3, 0.2, 0.3, 0.4], device="cuda:0")
        >>> ray_indices = torch.tensor([0, 0, 1, 1, 1], device="cuda:0")
        >>> def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        >>>     # This is a dummy function that returns random values.
        >>>     rgbs = torch.rand((t_starts.shape[0], 3), device="cuda:0")
        >>>     sigmas = torch.rand((t_starts.shape[0],), device="cuda:0")
        >>>     return rgbs, sigmas
        >>> colors, opacities, depths, extras = rendering(
        >>>     t_starts, t_ends, ray_indices, n_rays=2, rgb_sigma_fn=rgb_sigma_fn)
        >>> print(colors.shape, opacities.shape, depths.shape)
        torch.Size([2, 3]) torch.Size([2, 1]) torch.Size([2, 1])
        >>> extras.keys()
        dict_keys(['weights', 'alphas', 'trans'])

    """
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    if rgb_sigma_fn is None and rgb_alpha_fn is None:
        raise ValueError(
            "At least one of `rgb_sigma_fn` and `rgb_alpha_fn` should be specified."
        )

    # Query sigma/alpha and color with gradients
    if rgb_sigma_fn is not None:
        rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        # if t_starts.shape[0] != 0:
        #     rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
        # else:
        #     rgbs = torch.empty((0, 3), device=t_starts.device)
        #     sigmas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            sigmas.shape == t_starts.shape
        ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
        # Rendering: compute weights.
        weights, trans, alphas = nerfacc.volrend.render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "alphas": alphas,
            "trans": trans,
            "sigmas": sigmas,
            "rgbs": rgbs,
        }
    elif rgb_alpha_fn is not None:
        rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices)
        # if t_starts.shape[0] != 0:
        #     rgbs, alphas = rgb_alpha_fn(t_starts, t_ends, ray_indices)
        # else:
        #     rgbs = torch.empty((0, 3), device=t_starts.device)
        #     alphas = torch.empty((0,), device=t_starts.device)
        assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(
            rgbs.shape
        )
        assert (
            alphas.shape == t_starts.shape
        ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        # Rendering: compute weights.
        weights, trans = nerfacc.volrend.render_weight_from_alpha(
            alphas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        extras = {
            "weights": weights,
            "trans": trans,
            "rgbs": rgbs,
            "alphas": alphas,
        }

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = nerfacc.volrend.accumulate_along_rays(
        weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities = nerfacc.volrend.accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = nerfacc.volrend.accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    if expected_depths:
        depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, extras