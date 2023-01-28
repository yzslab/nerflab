import torch
from torch import nn
from internal.modules.encoding import get_encoding
from internal.modules.nerf_networks import get_nerf_network
from internal.nerf_sampling.common import generate_sample_output
from .rendering import *


def generate_coarse_sample_points(rays_o, rays_d, radii, near, far, n_samples, use_disp, perturb=1.):
    n_rays = near.shape[0]

    ## broadcast near far for each sample point
    near = torch.reshape(near, [-1, 1])
    far = torch.reshape(far, [-1, 1])

    t_vals = torch.linspace(0., 1., n_samples + 1, device=rays_o.device)

    if not use_disp:
        # not lindisp
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand(n_rays, n_samples + 1)

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

    t_mean, mean, cov_diag = get_cone_mean_conv(z_vals, rays_o, rays_d, radii)

    return z_vals, t_mean, mean, cov_diag


def sample_pdf(bins, weights, N_importance, alpha=1e-2, det=False):
    N_rays, N_samples_ = weights.shape
    weights_pad = torch.cat(
        [weights[..., :1], weights, weights[..., -1:]], dim=-1)
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    # prevent division by zero (don't do inplace op!)
    weights = weights + alpha
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1',
                           'sum')  # (N_rays, N_samples_)
    # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cumsum(pdf, -1)
    # (N_rays, N_samples_+1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance + 1, device=bins.device)
        u = u.expand(N_rays, N_importance + 1)
    else:
        u = torch.rand((N_rays, N_importance + 1), device=bins.device)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack(
        [below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled),
                      'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled),
                       'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < alpha] = 1  # denom equals 0 means a bin has weight 0,
    # in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / \
              denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def generate_fine_sample_points(rays_o, rays_d, radii, coarse_z_vals, coarse_weights, n_fine_samples, perturb):
    z_vals_mid = .5 * (coarse_z_vals[..., 1:] + coarse_z_vals[..., :-1])
    z_samples = sample_pdf(
        z_vals_mid, coarse_weights[..., 1:-1], n_fine_samples, det=(perturb == 0.))
    z_samples = z_samples.detach()

    # Obtain all points to evaluate color, density at.
    z_vals, _ = torch.sort(torch.concat([coarse_z_vals, z_samples], -1), -1)

    t_mean, mean, cov_diag = get_cone_mean_conv(z_vals, rays_o, rays_d, radii)

    return z_vals, t_mean, mean, cov_diag


class MipNeRFSampling(nn.Module):
    def __init__(
            self,
            hparams,
    ):
        super(MipNeRFSampling, self).__init__()
        self.hparams = hparams

        # create network input encoding
        # force IPE
        hparams["location_encoding"] = "ipe"
        if "ipe_location_n_freq" not in hparams:
            hparams["ipe_location_n_freq"] = 16
        self.location_encoder, self.view_direction_encoder = get_encoding(hparams)

        # coarse, N_sample
        # self.coarse_network = NeRFNetwork(**network_parameters)
        self.coarse_network = get_nerf_network(
            location_input_channels=self.location_encoder.get_output_n_channels(),
            view_direction_input_channels=self.view_direction_encoder.get_output_n_channels(),
            hparams=hparams,
        )

        # fine, N_importance
        if self.hparams["n_fine_samples"] > 0:
            # self.fine_network = NeRFNetwork(**network_parameters)
            self.fine_network = get_nerf_network(
                location_input_channels=self.location_encoder.get_output_n_channels(),
                view_direction_input_channels=self.view_direction_encoder.get_output_n_channels(),
                hparams=hparams,
            )

    def forward(
            self,
            rays_o,
            rays_d,
            view_directions,
            radii,
            near,
            far,
            n_coarse_samples,
            n_fine_samples,
            perturb: float,
            raw_noise_std: float,
    ):
        # coarse sample
        ## generate sample point
        coarse_z_vals, coarse_t_mean, coarse_mean, coarse_cov_diag = generate_coarse_sample_points(
            rays_o, rays_d,
            radii,
            near, far,
            n_coarse_samples,
            self.hparams["use_disp"],
            perturb,
        )
        ## query network by chunk
        coarse_network_output = self.run_network(
            network=self.coarse_network,
            view_directions=view_directions,
            mean=coarse_mean,
            cov_diag=coarse_cov_diag,
            chunk_size=self.hparams["chunk_size"],
        )
        ## convert network output to RGB
        coarse_rgb_map, coarse_disp_map, coarse_acc_map, coarse_weights, coarse_depth_map = raw2outputs(
            raw=coarse_network_output,
            z_vals=coarse_z_vals,
            mu_t=coarse_t_mean,
            raw_noise_std=raw_noise_std,
            white_bkgd=self.hparams["white_bkgd"],
        )

        results = {
            'coarse': generate_sample_output(coarse_network_output, coarse_rgb_map, coarse_disp_map, coarse_acc_map,
                                             coarse_weights, coarse_depth_map),
        }

        # fine sample
        if n_fine_samples > 0:
            fine_z_vals, fine_t_mean, fine_mean, fine_cov_diag = generate_fine_sample_points(
                rays_o, rays_d,
                radii,
                coarse_z_vals=coarse_z_vals,
                n_fine_samples=n_fine_samples,
                coarse_weights=coarse_weights,
                perturb=perturb,
            )

            fine_network_output = self.run_network(
                self.fine_network,
                view_directions,
                mean=fine_mean,
                cov_diag=fine_cov_diag,
                chunk_size=self.hparams["chunk_size"],
            )
            fine_rgb_map, fine_disp_map, fine_acc_map, fine_weights, fine_depth_map = raw2outputs(
                raw=fine_network_output,
                z_vals=fine_z_vals,
                mu_t=fine_t_mean,
                raw_noise_std=raw_noise_std,
                white_bkgd=self.hparams["white_bkgd"],
            )

            results['fine'] = generate_sample_output(fine_network_output, fine_rgb_map, fine_disp_map, fine_acc_map,
                                                     fine_weights,
                                                     fine_depth_map)

        return results

    def run_network(self, network, view_directions, mean, cov_diag, chunk_size):
        # encode location, [ray index][sample point index] = encoded xyz
        encoded_location = self.location_encoder(mean, cov_diag)
        # get shape to broadcast view direction
        sample_shape = list(encoded_location.shape[:2]) + [view_directions.shape[-1]]
        # reshape encoded location: [sample point index] = encoded feature vector
        encoded_location_flat = torch.reshape(encoded_location,
                                              (-1, self.location_encoder.get_output_n_channels()))

        # encode view direction, [ray index][sample point index] = encoded feature vector
        encoded_view_direction = self.view_direction_encoder(view_directions)
        # broadcast view direction, so every sample point has its own view direction
        encoded_view_direction = torch.broadcast_to(encoded_view_direction[:, None],
                                                    sample_shape[:-1] + [encoded_view_direction.shape[-1]])
        # reshape encoded view direction: [sample point index] = encoded view direction
        encoded_view_direction_flat = torch.reshape(encoded_view_direction, [-1, encoded_view_direction.shape[-1]])

        # concat encoded data: [sample point index] = encoded
        network_input_flat = torch.concat(
            [encoded_location_flat, encoded_view_direction_flat], -1)

        # query by chunk
        if chunk_size > 0:
            network_output_flat = []
            for i in range(0, network_input_flat.shape[0], chunk_size):
                network_output_flat.append(network(network_input_flat[i:i + chunk_size]))
            network_output_flat = torch.cat(network_output_flat, 0)
        else:
            network_output_flat = network(network_input_flat)

        # reshape back to index by [rays][sample points]
        network_output = torch.reshape(network_output_flat,
                                       sample_shape[:-1] + [network_output_flat.shape[-1]])
        return network_output
