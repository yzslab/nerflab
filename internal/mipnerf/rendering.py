import torch
from einops import rearrange, reduce, repeat


def get_cone_mean_conv(t_samples, rays_o, rays_d, radii):
    # t_samples:1024x65  rays_o:1024,3   radii:1024,1
    t0 = t_samples[..., :-1]  # left side
    t1 = t_samples[..., 1:]  # right side

    # conical_frustum_to_gaussian
    # gaussian approximation
    # eq-7
    t_mu = (t0 + t1) / 2
    t_sigma = (t1 - t0) / 2
    mu_t = t_mu + (2 * t_mu * t_sigma ** 2) / (3 * t_mu ** 2 + t_sigma ** 2)  # the real interval
    # 1024 x 64
    sigma_t = (t_sigma ** 2) / 3 - \
              (4 / 15) * \
              ((t_sigma ** 4 * (12 * t_mu ** 2 - t_sigma ** 2)) /
               (3 * t_mu ** 2 + t_sigma ** 2) ** 2)  # sigma_t
    sigma_r = radii ** 2 * \
              (
                      (t_mu ** 2) / 4 + (5 / 12) * t_sigma ** 2 - 4 /
                      15 * (t_sigma ** 4) / (3 * t_mu ** 2 + t_sigma ** 2)
              )

    # lift_gaussian()
    # calculate following the eq. 8
    # mean = torch.unsqueeze(rays_d, dim=-2) * torch.unsqueeze(mu_t, dim=-1)  # [B, 1, 3]*[B, N, 1] = [B, N, 3]
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')
    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    ## TODO: different with mip-nerf: mip.py:48
    mean = rays_o + rays_d * rearrange(mu_t, 'n1 n2 -> n1 n2 1')  # eq8
    # [1024,64,3]+[1024,1,3]*[1024,64,1]->[1024,64,3]
    # [B, 1, 3]*[B, N, 1] = [B, N, 3]

    rays_d = rays_d.squeeze()  # [1024,3]
    rays_o = rays_o.squeeze()  # [1024,3]
    # eq 16 mip-nerf
    dod = rays_d ** 2
    d2 = torch.sum(dod, dim=-1, keepdim=True) + 1e-10
    diagE = rearrange(sigma_t, 'n1 c -> n1 c 1') * rearrange(dod, 'n1 c -> n1 1 c') + \
            rearrange(sigma_r, 'n1 c -> n1 c 1') * \
            rearrange(1 - dod / d2, 'n1 c -> n1 1 c')

    return mu_t, mean, diagE  # [1024,64,3] [1024,64,3]


def raw2outputs(raw, z_vals, mu_t, raw_noise_std=0, white_bkgd=False):
    rgbs = raw[..., :3]
    sigmas = raw[..., 3]

    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    # delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) the last delta is infinity
    # deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)
    noise = 0.
    if raw_noise_std > 0:
        noise = torch.randn_like(sigmas)
    # (N_rays, N_samples_)
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1 -
                   alphas + 1e-10], -1)  # （1,1-a1,1-a2）
    Ti = torch.cumprod(alphas_shifted[:, :-1], -1)
    # cumprod: cumulated product, [:-1] is because eq3's upper index is i-1
    weights = alphas * Ti  # (N_rays, N_samples_)

    # (N_rays), the accumulated opacity along the rays
    weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')
    # ∑Ti*(1-exp(-δσ))
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    # results = {}
    # results['transmittance'] = Ti  # calculate the loss of the visibility network
    # results['weights'] = weights
    # results['opacity'] = weights_sum
    # results['z_vals'] = z_vals
    #
    # if type == "test_coarse":
    #     return results

    # weight:1024,64
    # rgb:1024,64,3
    # z_vals:1024,65
    rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')
                     * rgbs, 'n1 n2 c -> n1 c', 'sum')
    # depth_map = reduce(weights * z_vals, 'n1 n2 -> n1', 'sum')
    depth_map = reduce(weights * mu_t, 'n1 n2 -> n1', 'sum')
    if white_bkgd:
        rgb_map += 1 - weights_sum.unsqueeze(1)  # only needed in the white map

    return rgb_map, None, weights_sum, weights, depth_map
