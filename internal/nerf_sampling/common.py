import torch
import numpy as np


def generate_coarse_sample_points(rays_o, rays_d, near, far, n_samples, use_disp, perturb=1.):
    n_rays = near.shape[0]

    ## broadcast near far for each sample point
    near = torch.reshape(near, [-1, 1])
    far = torch.reshape(far, [-1, 1])

    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)

    if not use_disp:
        # not lindisp
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand(n_rays, n_samples)

    ## applying perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        # if pytest:
        #     np.random.seed(0)
        #     t_rand = np.random.rand(*list(z_vals.shape))
        #     t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    ## generating sample point coordinations
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    return pts, z_vals


def generate_fine_sample_points(rays_o, rays_d, coarse_z_vals, coarse_weights, n_fine_samples, perturb):
    z_vals_mid = .5 * (coarse_z_vals[..., 1:] + coarse_z_vals[..., :-1])
    z_samples = sample_pdf(
        z_vals_mid, coarse_weights[..., 1:-1], n_fine_samples, det=(perturb == 0.))
    z_samples = z_samples.detach()

    # Obtain all points to evaluate color, density at.
    z_vals, _ = torch.sort(torch.concat([coarse_z_vals, z_samples], -1), -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

    return pts, z_vals


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

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
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def generate_sample_output(raw, rgb_map, disp_map, acc_map, weights, depth_map):
    return {
        'raw': raw,
        'rgb_map': rgb_map,
        'disp_map': disp_map,
        'acc_map': acc_map,
        'weights': weights,
        'depth_map': depth_map,
    }
