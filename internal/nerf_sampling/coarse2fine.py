import torch
from torch import nn
from internal.rendering import raw2outputs
from internal.modules.encoding import get_encoding
from internal.modules.nerf_networks import get_nerf_network
from internal.nerf_sampling.common import generate_coarse_sample_points, generate_fine_sample_points, \
    generate_sample_output


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
        coarse_pts, coarse_z_vals = generate_coarse_sample_points(rays_o, rays_d, near, far, n_coarse_samples,
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
            fine_pts, fine_z_vals = generate_fine_sample_points(rays_o, rays_d, coarse_z_vals=coarse_z_vals,
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
