import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import line_profiler
# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.full((1,), 10., device=x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to16b = lambda x : ((2**16 - 1) * np.clip(x,0,1)).astype(np.uint16)

def precompute_quadratic_samples(near, far, num_samples):
    # normal parabola between 0.1 and 1, shifted and scaled to have y range between near and far
    start = 0.1
    x = torch.linspace(0, 1, num_samples)
    c = near
    a = (far - near)/(1. + 2. * start)
    b = 2. * start * a
    return a * x.pow(2) + b * x + c

def is_not_in_expected_distribution(depth_mean, depth_var, depth_measurement_mean, depth_measurement_std):
    delta_greater_than_expected = ((depth_mean - depth_measurement_mean).abs() - depth_measurement_std) > 0.
    var_greater_than_expected = depth_measurement_std.pow(2) < depth_var
    return torch.logical_or(delta_greater_than_expected, var_greater_than_expected)

# @line_profiler.profile
def compute_depth_loss(depth_map, s_val, target_depth, target_valid_depth, rays_d_norms):
    # print(target_valid_depth.dtype, target_valid_depth.shape)
    # nerfacc uses unormalized rays, we need to divide by rays_d_norms
    pred_mean = depth_map[target_valid_depth].squeeze(-1) / rays_d_norms[target_valid_depth].detach()
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    # divide variance by square of norms
    pred_var = s_val[target_valid_depth].squeeze(-1) / (rays_d_norms[target_valid_depth].detach() ** 2) + 1e-5

    target_mean = target_depth[..., 0][target_valid_depth]
    target_std = target_depth[..., 1][target_valid_depth]

    apply_depth_loss = is_not_in_expected_distribution(pred_mean, pred_var, target_mean, target_std)
    pred_mean = pred_mean[apply_depth_loss]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = pred_var[apply_depth_loss]
    target_mean = target_mean[apply_depth_loss]
    target_std = target_std[apply_depth_loss]
    f = torch.nn.GaussianNLLLoss(eps=0.001)
    return float(pred_mean.shape[0]) / float(target_valid_depth.shape[0]) * f(pred_mean, target_mean, pred_var)

class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * np.pi * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NerfMLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_cam=0, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NerfMLP, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_cam = input_ch_cam
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation="relu")] + [DenseLayer(W, W, activation="relu") if i not in self.skips else DenseLayer(W + input_ch, W, activation="relu") for i in range(D-1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([DenseLayer(input_ch_views + input_ch_cam + W, W//2, activation="relu")])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = DenseLayer(W, W, activation="linear")
            self.alpha_linear = DenseLayer(W, 1, activation="linear")
            self.rgb_linear = DenseLayer(W//2, 3, activation="linear")
        else:
            self.output_linear = DenseLayer(W, output_ch, activation="linear")

    def forward(self, x):

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views + self.input_ch_cam], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            sigma = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            # sigma = alpha.squeeze(-1)
        else:
            outputs = self.output_linear(h)
            rgb, sigma = outputs[..., :3], outputs[..., 3:]

        return rgb, sigma

    # Input: x are the positions with shape [n_samples, 64]
    # Output: densities must
    def query_density(self, x):
        # TODO: check if x input has the required shape
        # here x [n_samples, embed_dim] witoout view_dirs
        input_pts = x

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        sigma = self.alpha_linear(h)

        # TODO: check if raw densities has correct shape
        # raw_densities = torch.cat([outputs[..., :3], F.softplus(outputs[..., 3:], beta=10)], -1)


        return sigma


    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

def select_coordinates(coords, N_rand):
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords[select_inds].long()  # (N_rand, 2)
    return select_coords

def get_ray_dirs(H, W, intrinsic, c2w, coords=None):
    device = intrinsic.device
    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    if coords is None:
        i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
    else:
        i, j = coords[:, 1], coords[:, 0]
    # conversion from pixels to camera coordinates
    dirs = torch.stack([((i + 0.5)-cx)/fx, (H - (j + 0.5) - cy)/fy, -torch.ones_like(i)], -1) # center of pixel
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    return rays_d

# Ray helpers
def get_rays(H, W, intrinsic, c2w, coords=None):
    rays_d = get_ray_dirs(H, W, intrinsic, c2w, coords)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
