import os
import torch
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
import pytorch_lightning as pl
from internal.modules.nerf.nerf import NeRF as NeRFNetwork
import internal.rendering as rendering
from internal.dataset import extract_rays_data, extract_rays_rgb

from internal.modules.encoding import get_encoding
from internal.modules.loss.mse import MSELoss, img2mse, mse2psnr

import imageio
import numpy as np


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
        if self.hparams["n_fine_samples"] > 0:
            self.fine_network = NeRFNetwork(**network_parameters)

        # loss calculator
        self.loss_calculator = MSELoss()

    def forward(self, rays):
        """
        :param rays: see internals.dataset.NeRFDataset
        :return:
        """
        rays_o, rays_d, near, far, view_direction = extract_rays_data(rays)
        return self.coarse2fine_sample(rays_o=rays_o, rays_d=rays_d, view_directions=view_direction, near=near,
                                       far=far,
                                       n_coarse_samples=self.hparams["n_coarse_samples"],
                                       n_fine_samples=self.hparams["n_fine_samples"],
                                       perturb=self.hparams["perturb"])

    def training_step(self, batch, batch_idx):
        # rays_rgb = batch[2]
        rays_rgb = extract_rays_rgb(batch)
        # rendered_rays = self(batch["rays"])
        rendered_rays = self(batch)

        coarse_loss, fine_loss, coarse_psnr, fine_psnr = self.loss_calculator(rendered_rays, rays_rgb)
        loss = coarse_loss

        if self.hparams["n_fine_samples"] > 0:
            loss += fine_loss

        self.log("coarse/loss", coarse_loss, prog_bar=True)
        self.log("coarse/psnr", coarse_psnr, prog_bar=True)
        if self.hparams["n_fine_samples"] > 0:
            self.log("fine/loss", fine_loss, prog_bar=True)
            self.log("fine/psnr", fine_psnr, prog_bar=True)
        self.log("train/loss", loss)

        self.log("lrate", get_learning_rate(self.optimizer), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.render_single_image(batch)

    def on_predict_epoch_start(self):
        super().on_predict_epoch_start()
        self.val_save_dir = os.path.join(
            self.hparams["log_dir"],
            self.hparams["exp_name"],
            "val_images",
            f"step_{self.global_step}")
        os.makedirs(self.val_save_dir, exist_ok=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        predicted = self.render_single_image(batch)
        print(f"#{batch_idx} loss: {predicted['val/loss']}, psnr: {predicted['val/psnr'][0]}")
        imageio.imwrite(os.path.join(self.val_save_dir, '{:06d}.png'.format(batch_idx)),
                        to8b(predicted['val/img'].numpy()))
        return {
            "val/loss": predicted["val/loss"],
            "val/psnr": predicted["val/psnr"],
        }

    def configure_optimizers(self):
        lr = self.hparams["lrate"]
        eps = 1e-8

        parameters = list(self.coarse_network.parameters())
        if self.hparams["n_fine_samples"] > 0:
            parameters += list(self.fine_network.parameters())

        self.optimizer = torch.optim.Adam(parameters, lr=lr, eps=eps)

        # reduce learning rate
        # milestones = self.hparams["decay_step"]  # or 2, 4, 8 for blender
        # gamma = self.hparams["decay_gamma"]
        # scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)

        decay_rate = 0.1
        lrate_decay = self.hparams["lrate_decay"]
        decay_steps = lrate_decay * 1000

        def scheduler_func(step):
            return decay_rate ** (step / decay_steps)

        scheduler = LambdaLR(self.optimizer, scheduler_func)
        return [self.optimizer], [{
            "scheduler": scheduler,
            "interval": "step",
        }]

    def coarse2fine_sample(
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

        results = {
            'coarse': generate_sample_output(coarse_network_output, coarse_rgb_map, coarse_disp_map, coarse_acc_map,
                                             coarse_weights, coarse_depth_map),
        }

        # fine sample
        if n_fine_samples > 0:
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
        network_output_flat = []
        for i in range(0, network_input_flat.shape[0], chunk_size):
            network_output_flat.append(network(network_input_flat[i:i + chunk_size]))
        network_output_flat = torch.cat(network_output_flat, 0)

        # reshape back to index by [rays][sample points]
        network_output = torch.reshape(network_output_flat,
                                       list(pts.shape[:-1]) + [network_output_flat.shape[-1]])
        return network_output

    def render_single_image(self, batch):
        rays = batch["rays"][0]
        shape = batch["shape"]

        rendered_rays = {}
        for i in range(0, rays.shape[0], self.hparams["batch"]):
            rendered_batch_rays = self(rays[i:i + self.hparams["batch"]])['fine']
            for key in rendered_batch_rays:
                if key not in rendered_rays:
                    rendered_rays[key] = []
                rendered_rays[key].append(rendered_batch_rays[key])

        rendered_rays = {k: torch.concat(rendered_rays[k], 0) for k in rendered_rays}

        mse = img2mse(rendered_rays["rgb_map"], extract_rays_rgb(rays))
        psnr = mse2psnr(mse)

        # save image
        H, W = shape[0], shape[1]
        img = rendered_rays["rgb_map"].view(H, W, 3).cpu()
        # depth = rendering.visualize_depth(rendered_rays[f'depth_map'].view(H, W, 3))  # (3, H, W)

        return {
            "val/loss": mse,
            "val/psnr": psnr,
            "val/img": img,
            # "val/depth": depth,
        }


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


def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)
