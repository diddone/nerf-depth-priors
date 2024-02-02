import torch
from torch import nn
from new_model.mlp import NerfMLP, get_embedder
import torch.nn.functional as F




class VanillaNeRFRadianceField(nn.Module):
    def __init__(
        self,
        trainset_length, # len of training set for camera embeddings
        args, # here we have parameters for building nerf
        device
    ) -> None:
        super().__init__()

        self.args = args
        self.device = device
        # embeddings for the camera positions, we need so set current camera position later
        self.embedcam_fn = torch.nn.Embedding(trainset_length, args.input_ch_cam)

        # embeddings for the positions of given point and its output dimension
        embedpos_fn, input_ch = get_embedder(args.multires, args.i_embed)
        self.embedpos_fn = embedpos_fn

         # create embeddings for the directions the same way as for the positions
        input_ch_views = 0
        embeddirs_fn = None
        if args.use_viewdirs:
            # this should be none? (we dont use embeddings for the view dir)
            embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
            self.embeddirs_fn = embeddirs_fn
        output_ch = 5 if args.N_importance > 0 else 4
        skips = [4]

        self.mlp = NerfMLP(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)

        self.camera_idx = None

    # use this functions in the training to set camera_idx
    # TODO add to the train loop
    def set_camera_idx(self, idx):
        self.camera_idx = idx
        self.embedded_cam = self.embedcam_fn(torch.tensor(idx, device=self.device))

    def set_camera_embed_zeros(self):
        self.embedded_cam = torch.zeros((self.args.input_ch_cam), device=self.device)

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    # NerfAcc function:
    # Input: x has shape [n_samples, 3]
    # Output: sigmas has shape [n_samples]
    def query_density(self, x):
        x = self.embed_input(x)
        sigma = self.mlp.query_density(x)
        return F.softplus(sigma, beta=10)

    # (n_samples, 3) -> (n_samples, 57) if NOT using vierwdirs
    def embed_input(self, inputs, t_dirs=None):
        # (n_samples, 3) -> (n_samples, embed_dim)
        embedded = self.embedpos_fn(inputs) # n_samples, multires * 2 * 3 + 3

        if t_dirs is not None:
            if self.camera_idx is None:
                raise ValueError("Camera idx is None, but should be set for camera embeddings")
            # this should be no-op, since we are not using embeddings for directions
            embedded_dirs = self.embeddirs_fn(t_dirs)
            embedded_cam = self.embedded_cam
            embedded = torch.cat(
                [embedded, embedded_dirs,
                embedded_cam.unsqueeze(0).expand(embedded_dirs.shape[0], embedded_cam.shape[0]
            )], -1)

        return embedded
        # outputs_flat = batchify(fn, netchunk)(embedded)

    def forward(self, x, t_dirs=None):
        # we encode view info also there
        x = self.embed_input(x, t_dirs=t_dirs)
        # print("x shape to mlp", x.shape)
        rgb, sigma = self.mlp(x)
        return torch.sigmoid(rgb), F.softplus(sigma, beta=10)
    # (n_samples, 3 + conditions) -> (n_samples, 64) if using vierwdirs