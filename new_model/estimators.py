from abc import ABC, abstractmethod
import torch
from torch import nn
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.estimators.prop_net import PropNetEstimator

def build_estimator(args):
    if args.estim_type == "default":
       return OccGridEstimator(args.aabb, args.occ_resolution, args.occ_num_levels)
    elif args.estim_type == "occgrid":
       return OccGridEstim(args.aabb, args.occ_resolution, args.occ_num_levels)
    elif args.estim_type == "propnet":
       raise NotImplementedError
    else:
        raise ValueError(f"Invalid estimator type {args.estim_type}, available types are 'occgrid' and 'propnet'")

class Estimator(ABC):

    # @staticmethod
    # @abstractmethod
    # def sigma_fn():
    #   """
    #   Abstract method to be implemented by subclasses.
    #   """
    #   pass

    # @staticmethod
    # @abstractmethod
    # def rgb_sigma_fn():
    #   """
    #   Abstract method to be implemented by subclasses.
    #   """
    #   pass

    # @abstractmethod
    # def sampling(self):
    #   pass

    @abstractmethod
    def update_every_n_steps(self):
      """
      Abstract method to be implemented by subclasses.
      """
      pass

    def set_rays(self, rays_o, rays_d):
       self.rays_o = rays_o
       self.rays_d = rays_d

class OccGridEstim(nn.Module, Estimator):

  def __init__(self, aabb, resolution, lvls, occ_thres=1e-2, warmup_steps=128):
    super(OccGridEstim, self).__init__()
    self.occ_grid_estimator = OccGridEstimator(aabb, resolution, lvls)
    self.occ_thres = occ_thres
    self.warmup_steps = warmup_steps

  def update_every_n_steps(self, step, transm, occ_eval_fn):
    """
    Updates the occ_grid_estimator every n steps.

    Args:
      step (int): The current step.
      transmittance: required for pronet.
      occ_eval_fn: The occ_eval_fn object.

    Returns:
      None
    """
    self.occ_grid_estimator.update_every_n_steps(
      step=step,
      occ_eval_fn=occ_eval_fn,
      occ_thre=self.occ_thres,
      warmup_steps=self.warmup_steps
    )

  @staticmethod
  def sigma_fn(_rays_o, _rays_d, _radiance_field, t_starts, t_ends, ray_indices):
    t_origins = _rays_o[ray_indices]
    t_dirs = _rays_d[ray_indices]
    positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

    sigmas = _radiance_field.query_density(positions)
    return sigmas.squeeze(-1)

  @staticmethod
  def rgb_sigma_fn(_rays_o, _rays_d, _radiance_field, t_starts, t_ends, ray_indices):
      t_origins = _rays_o[ray_indices]
      t_dirs = _rays_d[ray_indices]
      positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

      rgbs, sigmas = _radiance_field(positions, t_dirs)
      return rgbs, sigmas.squeeze(-1)

  def sampling(
        self, rays_o, rays_d, sigma_fn,
        near_plane, far_plane, render_step_size, stratified,
        cone_angle=0.0, alpha_thre=0.0,
    ):
      return self.occ_grid_estimator.sampling(
        rays_o=rays_o,
        rays_d=rays_d,
        sigma_fn=sigma_fn,
        near_plane=near_plane,
        far_plane=far_plane,
        render_step_size=render_step_size,
        stratified=stratified,
        cone_angle=cone_angle,
        alpha_thre=alpha_thre,
      )


class PropNetEstim(nn.Module, Estimator):
  def __init__(self, optimizer, scheduler,):
    super(PropNetEstim, self).__init__()
    self.prop_net_estimator = PropNetEstimator(optimizer, scheduler)






