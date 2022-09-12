import torch
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
from internal.modules.nerf.nerf import NeRF as NeRFNetwork
import internal.rendering as rendering
from internal.dataset import extract_rays_data, extract_rays_rgb

from internal.modules.encoding import get_encoding
from internal.modules.loss.mse import MSELoss


class NeRF(pl.LightningModule):
    def __init__(
            self,
            hparams,
    ):
        super().__init__()
        self.save_hyperparameters(hparams),

        # create network input encoding
        self.location_encoder, self.view_direction_encoder = get_encoding(hparams)

        # network parameters
        network_parameters = {
            "density_layers": hparams["density_layers"],
            "density_layer_units": hparams["density_layer_units"],
            "color_layers": hparams["color_layers"],
            "color_layer_units": hparams["color_layer_units"],
            "skips": hparams["skips"],
            "location_input_channels": self.location_encoder.get_output_n_channels(),
            "view_direction_input_channels": self.view_direction_encoder.get_output_n_channels(),
        }

        # coarse, N_sample
        self.coarse_network = NeRFNetwork(**network_parameters)

        # fine, N_importance
        self.fine_network = NeRFNetwork(**network_parameters)

        # loss calculator
        self.loss_calculator = MSELoss()

    def forward(self, rays):
        """
        :param rays: see internals.dataset.NeRFDataset
        :return:
        """
        rays_o, rays_d, near, far, view_direction = extract_rays_data(rays)
        return self.coarse_2_fine_sample(rays_o=rays_o, rays_d=rays_d, view_directions=view_direction, near=near,
                                         far=far,
                                         n_coarse_samples=self.hparams["n_coarse_samples"],
                                         n_fine_samples=self.hparams["n_fine_samples"],
                                         perturb=self.hparams["perturb"])

    def training_step(self, batch, batch_idx):
        # rays_rgb = batch[2]
        rays_rgb = extract_rays_rgb(batch)
        rendered_rays = self(batch["rays"])

        coarse_loss, fine_loss, coarse_psnr, fine_psnr = self.loss_calculator(rendered_rays, rays_rgb)
        loss = coarse_loss + fine_loss

        self.log("coarse/loss", coarse_loss, prog_bar=True)
        self.log("coarse/psnr", coarse_psnr, prog_bar=True)
        self.log("fine/loss", fine_loss, prog_bar=True)
        self.log("fine/psnr", fine_psnr, prog_bar=True)
        self.log("train/loss", loss)

        self.log("lrate", get_learning_rate(self.optimizer), prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = self.hparams["lrate"]
        eps = 1e-8

        parameters = list(self.coarse_network.parameters())
        parameters += list(self.fine_network.parameters())

        self.optimizer = torch.optim.Adam(parameters, lr=lr, eps=eps)

        # reduce learning rate
        milestones = self.hparams["decay_step"]  # or 2, 4, 8 for blender
        gamma = self.hparams["decay_gamma"]
        scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        return [self.optimizer], [scheduler]

    def coarse_2_fine_sample(
            self,
            rays_o,
            rays_d,
            view_directions,
            near,
            far,
            n_coarse_samples,
            n_fine_samples,
            perturb=1.
    ):
        # coarse sample
        coarse_pts, coarse_z_vals = rendering.generate_coarse_sample_points(rays_o, rays_d, near, far, n_coarse_samples,
                                                                            perturb)
        coarse_network_output = self.run_network(self.coarse_network, coarse_pts, view_directions,
                                                 self.hparams["chunk"])
        coarse_rgb_map, coarse_disp_map, coarse_acc_map, coarse_weights, coarse_depth_map = rendering.raw2outputs(
            raw=coarse_network_output,
            z_vals=coarse_z_vals,
            rays_d=rays_d,
            raw_noise_std=self.hparams["noise_std"],
            white_bkgd=self.hparams["white_background"]
        )

        # fine sample
        fine_pts, fine_z_vals = rendering.generate_fine_sample_points(rays_o, rays_d, coarse_z_vals=coarse_z_vals,
                                                                      n_fine_samples=n_fine_samples,
                                                                      coarse_weights=coarse_weights,
                                                                      perturb=perturb)
        fine_network_output = self.run_network(self.fine_network, fine_pts, view_directions, self.hparams["chunk"])
        fine_rgb_map, fine_disp_map, fine_acc_map, fine_weights, fine_depth_map = rendering.raw2outputs(
            raw=fine_network_output,
            z_vals=fine_z_vals,
            rays_d=rays_d,
            raw_noise_std=self.hparams["noise_std"],
            white_bkgd=self.hparams["white_background"]
        )

        return {
            'coarse': generate_sample_output(coarse_network_output, coarse_rgb_map, coarse_disp_map, coarse_acc_map,
                                             coarse_weights, coarse_depth_map),
            'fine': generate_sample_output(fine_network_output, fine_rgb_map, fine_disp_map, fine_acc_map, fine_weights,
                                           fine_depth_map),
        }

    def run_network(self, network, pts, view_directions, chunk_size):
        # index by sample point
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        view_directions = torch.broadcast_to(view_directions[:, None], pts.shape)
        view_directions_flat = torch.reshape(view_directions, [-1, view_directions.shape[-1]])
        network_input_flat = torch.concat(
            [self.location_encoder(pts_flat), self.view_direction_encoder(view_directions_flat)], -1)

        # query by chunk
        network_output_flat = []
        for i in range(0, network_input_flat.shape[0], chunk_size):
            network_output_flat.append(network(network_input_flat[i:i + chunk_size]))
        network_output_flat = torch.cat(network_output_flat, 0)

        # reshape back to index by [rays][sample points]
        network_output = torch.reshape(network_output_flat,
                                       list(pts.shape[:-1]) + [network_output_flat.shape[-1]])
        return network_output


def generate_sample_output(raw, rgb_map, disp_map, acc_map, weights, depth_map):
    return {
        'raw': raw,
        'rgb_map': rgb_map,
        'disp_map': disp_map,
        'acc_map': acc_map,
        'weights': weights,
        'depth_map': depth_map,
    }


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
