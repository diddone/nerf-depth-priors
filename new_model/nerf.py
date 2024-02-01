import torch
from torch import nn
from mlp import NerfMLP, get_embedder


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # use embed
    embedpos_fn = get_embedder_cls(args.multires, args.i_embed)
    input_ch = embedpos_fn.out_dim
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)

    return model

class VanillaNeRFRadianceField(nn.Module):
    def __init__(
        self,
        i_train, # len of training set for camera embeddings
        args, # here we have parameters for buikding nerf
    ) -> None:
        super().__init__()
        # embeddings for the camera positions, we need so set current camera position later
        self.embedcam_fn = torch.nn.Embedding(len(i_train), input_ch_cam)

        # embeddings for the positions of given point and its output dimension
        embedpos_fn, input_ch = get_embedder(args.multires, args.i_embed)
        self.embedpos_fn = embedpos_fn

         # create embeddings for the directions the same way as for the positions
        input_ch_views = 0
        embeddirs_fn = None
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
            self.embeddirs_fn = embeddirs_fn
        output_ch = 5 if args.N_importance > 0 else 4
        skips = [4]

        self.mlp = NerfMLP(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x):
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x)
        return F.relu(sigma)

    def forward(self, x, condition=None):
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition)
        return torch.sigmoid(rgb), F.relu(sigma)