import torch
from torch import nn
from internal.rendering import raw2outputs
import numpy as np
from internal.modules.encoding import get_encoding
from internal.modules.nerf_networks import get_nerf_network
from internal.nerf_sampling.common import generate_sample_output


class Coarse2Fine(nn.Module):
    def __init__(
            self,
            hparams,
    ):
        super(Coarse2Fine, self).__init__()
        self.hparams = hparams

        # create network input encoding
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
            near,
            far,
            n_coarse_samples,
            n_fine_samples,
            perturb: float,
            raw_noise_std: float,
    ):
        def clip_pts(pts):
            pts = torch.min(
                torch.max(
                    pts,
                    torch.tensor(self.hparams["bounding_box"][0], device=pts.device),
                ),
                torch.tensor(self.hparams["bounding_box"][1], device=pts.device)
            )  # a manual clip.
            return pts

        # coarse sample
        coarse_pts, coarse_z_vals = self.generate_coarse_sample_points(rays_o, rays_d, near, far, n_coarse_samples,
                                                                       self.hparams["use_disp"], perturb)
        if "network_type" in self.hparams and self.hparams["network_type"] == "tcnn_ff":
            coarse_pts = clip_pts(coarse_pts)
            sample_dist = (far - near) / n_coarse_samples
        else:
            sample_dist = None

        coarse_network_output = self.run_network(self.coarse_network, coarse_pts, view_directions,
                                                 self.hparams["chunk_size"])
        coarse_rgb_map, coarse_disp_map, coarse_acc_map, coarse_weights, coarse_depth_map = raw2outputs(
            raw=coarse_network_output,
            z_vals=coarse_z_vals,
            rays_d=rays_d,
            raw_noise_std=raw_noise_std,
            white_bkgd=self.hparams["white_bkgd"],
            sample_dist=sample_dist,
        )

        results = {
            'coarse': generate_sample_output(coarse_network_output, coarse_rgb_map, coarse_disp_map, coarse_acc_map,
                                             coarse_weights, coarse_depth_map),
        }

        # fine sample
        if n_fine_samples > 0:
            fine_pts, fine_z_vals = self.generate_fine_sample_points(rays_o, rays_d, coarse_z_vals=coarse_z_vals,
                                                                     n_fine_samples=n_fine_samples,
                                                                     coarse_weights=coarse_weights,
                                                                     perturb=perturb)
            if "network_type" in self.hparams and self.hparams["network_type"] == "tcnn_ff":
                fine_pts = clip_pts(fine_pts)

            fine_network_output = self.run_network(self.fine_network, fine_pts, view_directions,
                                                   self.hparams["chunk_size"])
            fine_rgb_map, fine_disp_map, fine_acc_map, fine_weights, fine_depth_map = raw2outputs(
                raw=fine_network_output,
                z_vals=fine_z_vals,
                rays_d=rays_d,
                raw_noise_std=raw_noise_std,
                white_bkgd=self.hparams["white_bkgd"],
                sample_dist=sample_dist,
            )

            results['fine'] = generate_sample_output(fine_network_output, fine_rgb_map, fine_disp_map, fine_acc_map,
                                                     fine_weights,
                                                     fine_depth_map)

        return results

    def run_network(self, network, pts, view_directions, chunk_size):
        # index by sample point
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        view_directions = torch.broadcast_to(view_directions[:, None], pts.shape)
        view_directions_flat = torch.reshape(view_directions, [-1, view_directions.shape[-1]])
        network_input_flat = torch.concat(
            [self.location_encoder(pts_flat), self.view_direction_encoder(view_directions_flat)], -1)

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
                                       list(pts.shape[:-1]) + [network_output_flat.shape[-1]])
        return network_output

    def generate_coarse_sample_points(self, rays_o, rays_d, near, far, n_samples, use_disp, perturb=1.):
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

    def generate_fine_sample_points(self, rays_o, rays_d, coarse_z_vals, coarse_weights, n_fine_samples, perturb):
        z_vals_mid = .5 * (coarse_z_vals[..., 1:] + coarse_z_vals[..., :-1])
        z_samples = self.sample_pdf(
            z_vals_mid, coarse_weights[..., 1:-1], n_fine_samples, det=(perturb == 0.))
        z_samples = z_samples.detach()

        # Obtain all points to evaluate color, density at.
        z_vals, _ = torch.sort(torch.concat([coarse_z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        return pts, z_vals

    # Hierarchical sampling (section 5.2)
    def sample_pdf(self, bins, weights, N_samples, det=False, pytest=False):
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
