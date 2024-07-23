from typing import Union, List, Callable
import torch
from torch import nn
from new_model.mlp import NerfMLP, get_embedder
from abc import ABC, abstractmethod
import torch.nn.functional as F

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


def build_radiance_field(trainset_length, args, device):
    if args.model_type == "original":
        return VanillaNeRFRadianceField(trainset_length, args, device)
    elif args.model_type == "ngp":
        return NGPRadianceField(trainset_length, args, device, max_resolution=2048 * 8) # 2048 * scene size
    else:
        raise ValueError(f"Unknown model type: {args.model_type}, available options are: original, ngp)")

class RadianceFieldBase(nn.Module, ABC):

    def __init__(self, trainset_length, args, device):
        super().__init__()
        self.embedcam_fn = torch.nn.Embedding(trainset_length, args.input_ch_cam)
        self.input_ch_cam = args.input_ch_cam
        self.device = device

        self.camera_idx = None
        self.embedded_cam = None

        input_ch_views = 0
        embeddirs_fn = None
        if args.use_viewdirs:
            # this should be none? (we dont use embeddings for the view dir)
            embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
            self.embeddirs_fn = embeddirs_fn
        self.input_ch_views = input_ch_views


    def set_camera_idx(self, idx):
        self.camera_idx = idx
        self.embedded_cam = self.embedcam_fn(torch.tensor(idx, device=self.device))

    def set_camera_embed_zeros(self):
        self.embedded_cam = torch.zeros((self.input_ch_cam), device=self.device)

    def embed_cam_and_dirs(self, embedded, t_dirs):
        # assert t_dirs is not None
        if t_dirs is not None:
            if self.embedded_cam is None:
                raise ValueError("Camera embed is None, but should be set for camera embeddings")
            # this should be no-op, since we are not using embeddings for directions
            embedded_dirs = self.embeddirs_fn(t_dirs)
            embedded_cam = self.embedded_cam
            embedded = torch.cat(
                [embedded, embedded_dirs,
                embedded_cam.unsqueeze(0).expand(embedded_dirs.shape[0], embedded_cam.shape[0]
            )], -1)
        return embedded

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the stepsize is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    @abstractmethod
    def query_density(self, x):
        pass



class VanillaNeRFRadianceField(RadianceFieldBase):
    def __init__(
        self,
        trainset_length, # len of training set for camera embeddings
        args, # here we have parameters for building nerf
        device
    ) -> None:
        super().__init__(trainset_length, args, device)

        self.args = args
        self.device = device

        # embeddings for the camera positions, we need so set current camera position later
        # self.embedcam_fn = torch.nn.Embedding(trainset_length, args.input_ch_cam)

        # embeddings for the positions of given point and its output dimension
        embedpos_fn, input_ch = get_embedder(args.multires, args.i_embed)
        self.embedpos_fn = embedpos_fn

         # create embeddings for the directions the same way as for the positions

        output_ch = 5 if args.N_importance > 0 else 4
        skips = [4]

        self.mlp = NerfMLP(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=self.input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)


    # # use this functions in the training to set camera_idx
    # # TODO add to the train loop
    # def set_camera_idx(self, idx):
    #     self.camera_idx = idx
    #     self.embedded_cam = self.embedcam_fn(torch.tensor(idx, device=self.device))

    # def set_camera_embed_zeros(self):
    #     self.embedded_cam = torch.zeros((self.args.input_ch_cam), device=self.device)


    # NerfAcc function:
    # Input: x has shape [n_samples, 3]
    # Output: sigmas has shape [n_samples]

    # (n_samples, 3) -> (n_samples, 57) if NOT using vierwdirs
    def embed_input(self, inputs):
        inputs = (inputs-self.args.bb_center)*self.args.bb_scale
        embedded = self.embedpos_fn(inputs)
        return embedded

    def prepare_input(self, inputs, t_dirs=None):
        # (n_samples, 3) -> (n_samples, embed_dim)
        inputs = self.embed_input(inputs) # [N, 3] -> [N, multires * 2 * 3 + 3]
        embedded = self.embed_cam_and_dirs(inputs, t_dirs)

        return embedded

    def query_density(self, x):
        x = self.prepare_input(x)
        sigma = self.mlp.query_density(x)
        return F.softplus(sigma, beta=10)

    def forward(self, x, t_dirs=None):
        # we encode view info also there
        x = self.prepare_input(x, t_dirs=t_dirs)
        # print("x shape to mlp", x.shape)
        rgb, sigma = self.mlp(x)
        return torch.sigmoid(rgb), F.softplus(sigma, beta=10)
    # (n_samples, 3 + conditions) -> (n_samples, 64) if using vierwdirs


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply

class NGPRadianceField(RadianceFieldBase):
    """Instance-NGP Radiance Field"""

    def __init__(
        self,
        trainset_length, args, device,
        # aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = False,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        base_resolution: int = 16,
        max_resolution: int = 4096,
        geo_feat_dim: int = 15,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
    ) -> None:
        super().__init__(trainset_length, args, device)
        aabb = args.aabb
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)

        # Turns out rectangle aabb will leads to uneven collision so bad performance.
        # We enforce a cube aabb here.
        center = (aabb[..., :num_dim] + aabb[..., num_dim:]) / 2.0
        size = (aabb[..., num_dim:] - aabb[..., :num_dim]).max()
        aabb = torch.cat([center - size / 2.0, center + size / 2.0], dim=-1)

        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation

        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.geo_feat_dim = geo_feat_dim
        self.n_levels = n_levels
        self.log2_hashmap_size = log2_hashmap_size

        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                        # {"otype": "Identity", "n_bins": 4, "degree": 4},
                    ],
                },
            )

        mlp_otype = "CutlassMLP" if args.netwidth not in [32, 64, 128] else "FullyFusedMLP"
        self.mlp_base = tcnn.NetworkWithInputEncoding(
            n_input_dims=num_dim,
            n_output_dims=1 + self.geo_feat_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": args.log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            network_config={
                "otype": mlp_otype,
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": args.netwidth,
                "n_hidden_layers": 1,
            },
        )
        if self.geo_feat_dim > 0:
            self.mlp_head = tcnn.Network(
                n_input_dims=(
                    self.input_ch_views + self.input_ch_cam + self.geo_feat_dim
                ),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": args.netwidth,
                    "n_hidden_layers": 3,
                },
            )

    def embed_input(self, inputs):
        aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
        inputs = (inputs - aabb_min) / (aabb_max - aabb_min)

        return self.embed_fn(inputs)

    def query_density(self, x, return_feat: bool = False):
        aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        # selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        x = (
            self.mlp_base(x.view(-1, self.num_dim))
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )
        density_before_activation, base_mlp_out = torch.split(
            x, [1, self.geo_feat_dim], dim=-1
        )
        density = (
            self.density_activation(density_before_activation)
            # * selector[..., None]
        )
        if return_feat:
            return density, base_mlp_out
        else:
            return density

    def _query_rgb(self, dir, embedding, apply_act: bool = True):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.reshape(-1, dir.shape[-1]))
            h = torch.cat([d, embedding.reshape(-1, self.geo_feat_dim)], dim=-1)
        else:
            h = self.embed_cam_and_dirs(embedding.reshape(-1, self.geo_feat_dim), t_dirs=dir)
            # h = embedding.reshape(-1, self.geo_feat_dim)
        rgb = (
            self.mlp_head(h)
            .reshape(list(embedding.shape[:-1]) + [3])
            .to(embedding)
        )
        if apply_act:
            rgb = torch.sigmoid(rgb)
        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"
        density, embedding = self.query_density(positions, return_feat=True)
        rgb = self._query_rgb(directions, embedding=embedding)
        return rgb, density  # type: ignore
