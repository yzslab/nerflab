import os
import torch
import pytorch_lightning as pl
import yaml
import internal.optimizer
import internal.rendering as rendering
from kornia.losses import ssim_loss
from internal.dataset import extract_rays_data, extract_rays_rgb
from internal.modules.loss.mse import MSELoss, img2mse, mse2psnr
import imageio
import numpy as np
import internal.nerf_sampling.coarse2fine


# import internal.nerf_sampling.coarse2fine_single_network


class NeRF(pl.LightningModule):
    def __init__(
            self,
            hparams,
    ):
        super().__init__()
        self.save_hyperparameters(hparams),

        sample_type = hparams["sample_type"]
        if sample_type == "coarse2fine":
            self.sampling = internal.nerf_sampling.coarse2fine.Coarse2Fine(hparams)
        # elif sample_type == "coarse2fine_sn":
        #     self.sampling = internal.nerf_sampling.coarse2fine_single_network.Coarse2FineSingleNetwork(hparams)
        else:
            raise ValueError(f"unsupported sample type: {sample_type}")

        # loss calculator
        self.loss_calculator = MSELoss()

    def forward(self, rays, perturb: float, raw_noise_std: float):
        """
        :param rays: see internals.dataset.NeRFDataset
        :param perturb
        :param raw_noise_std
        :return:
        """
        rays_o, rays_d, near, far, view_direction = extract_rays_data(rays)
        return self.sampling(rays_o=rays_o, rays_d=rays_d, view_directions=view_direction, near=near,
                             far=far,
                             n_coarse_samples=self.hparams["n_coarse_samples"],
                             n_fine_samples=self.hparams["n_fine_samples"],
                             perturb=perturb,
                             raw_noise_std=raw_noise_std,
                             )

    def training_step(self, batch, batch_idx):
        # rays_rgb = batch[2]
        rays_rgb = extract_rays_rgb(batch)
        # rendered_rays = self(batch["rays"])
        rendered_rays = self(batch, self.hparams["perturb"], self.hparams["noise_std"])

        coarse_loss, fine_loss, coarse_psnr, fine_psnr = self.loss_calculator(rendered_rays, rays_rgb)
        loss = coarse_loss

        if self.hparams["n_fine_samples"] > 0:
            loss += fine_loss

        self.log("coarse/loss", coarse_loss, prog_bar=False)
        if self.hparams["n_fine_samples"] > 0:
            self.log("coarse/psnr", coarse_psnr, prog_bar=False)
            self.log("fine/loss", fine_loss, prog_bar=False)
            self.log("fine/psnr", fine_psnr, prog_bar=True)
        else:
            self.log("coarse/psnr", coarse_psnr, prog_bar=True)

        self.log("train/loss", loss)

        self.log("lrate", get_learning_rate(self.optimizer), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        rendered = self.render_single_image(batch)

        # only log the first result to logger
        if batch_idx == 0:
            img = rendered["val/img"].permute(2, 0, 1)
            gt = rendered["val/gt"].permute(2, 0, 1)
            stack = torch.stack([gt, img, rendered["val/depth_chw"]])
            self.logger.experiment.add_images("val/gt_pred_depth", stack, self.global_step)

        return {
            "val/loss": rendered["val/loss"],
            "val/psnr": rendered["val/psnr"],
            "val/ssim": rendered["val/ssim"],
        }

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val/loss"] for x in outputs]).mean()
        mean_acc = torch.stack([x["val/psnr"] for x in outputs]).mean()
        mean_ssim = torch.stack([x["val/ssim"] for x in outputs]).mean()

        self.log("val/loss", mean_loss, prog_bar=True)
        self.log("val/psnr", mean_acc, prog_bar=True)
        self.log("val/ssim", mean_ssim, prog_bar=True)

    def on_predict_epoch_start(self):
        self.val_save_dir = os.path.join(
            self.hparams["log_dir"],
            self.hparams["exp_name"],
            "val_images",
            self.hparams["eval_name"])
        os.makedirs(self.val_save_dir, exist_ok=True)

        # dump hyperparameter to yaml file
        with open(os.path.join(self.val_save_dir, "hparams.yaml"), "w") as f:
            yaml.dump(dict(self.hparams), f)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # render an image
        predicted = self.render_single_image(batch)
        print(f"#{batch_idx} loss: {predicted['val/loss']}, psnr: {predicted['val/psnr'][0]}, ssim: {predicted['val/ssim']}")

        output_filename = "{:06d}".format(batch_idx)
        if "filename" in batch:
            output_filename = batch["filename"][0]
            os.makedirs(os.path.join(self.val_save_dir, os.path.dirname(output_filename)), exist_ok=True)

        # save rgb
        imageio.imwrite(os.path.join(self.val_save_dir, '{}_rgb.png'.format(output_filename)),
                        to8b(predicted['val/img'].numpy()))
        # save depth map
        depth_map = predicted["val/depth_map"]
        imageio.imwrite(os.path.join(self.val_save_dir, '{}_depth.png'.format(output_filename)),
                        depth_map)

        return {
            "id": batch_idx,
            "val/loss": predicted["val/loss"],
            "val/psnr": predicted["val/psnr"][0],
            "val/ssim": predicted["val/ssim"],
            "val/img": predicted["val/img"],
        }

    def on_predict_epoch_end(self, results):
        loss_values = []
        psnr_values = []
        ssim_values = []
        with open(os.path.join(self.val_save_dir, "metrics.txt"), mode="w") as f:
            for i in results:
                for image in i:
                    loss_values.append(image["val/loss"])
                    psnr_values.append(image["val/psnr"])
                    ssim_values.append(image["val/ssim"])
                    f.write(f"#{image['id']} loss: {image['val/loss']}, psnr: {image['val/psnr']}, ssim: {image['val/ssim']}\n")
            mean_loss = torch.tensor(loss_values).mean()
            mean_psnr = torch.tensor(psnr_values).mean()
            mean_ssim = torch.tensor(ssim_values).mean()

            mean_text = f"mean: loss: {mean_loss}, psnr: {mean_psnr}, ssim: {mean_ssim}"
            f.write(mean_text)
            f.write("\n")
            print(mean_text)

    def configure_optimizers(self):
        models = [self.sampling]
        self.optimizer = internal.optimizer.get_optimizer(models, self.hparams)
        scheduler = internal.optimizer.get_scheduler(self.optimizer, self.hparams)

        return [self.optimizer], [scheduler]
        # lr = self.hparams["lrate"]
        # eps = 1e-8
        #
        # parameters = list(self.coarse_network.parameters())
        # if self.hparams["n_fine_samples"] > 0:
        #     parameters += list(self.fine_network.parameters())
        #
        # self.optimizer = torch.optim.Adam(parameters, lr=lr, eps=eps)
        #
        # # reduce learning rate
        # # milestones = self.hparams["decay_step"]  # or 2, 4, 8 for blender
        # # gamma = self.hparams["decay_gamma"]
        # # scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        #
        # decay_rate = 0.1
        # lrate_decay = self.hparams["lrate_decay"]
        # decay_steps = lrate_decay * 1000
        #
        # def scheduler_func(step):
        #     return decay_rate ** (step / decay_steps)
        #
        # scheduler = LambdaLR(self.optimizer, scheduler_func)
        # return [self.optimizer], [{
        #     "scheduler": scheduler,
        #     "interval": "step",
        # }]

    def render_single_image(self, batch):
        shape = batch["shape"]

        render_result_key = "coarse"
        if self.hparams["n_fine_samples"] > 0:
            render_result_key = "fine"

        rendered_rays = {}
        for i in range(0, batch["rays"][0][0].shape[0], self.hparams["batch_size"]):
            # build batchify ray
            batchify_rays = []
            for ray_data_index in range(0, len(batch["rays"])):
                ray_data = batch["rays"][ray_data_index][0]
                batchify_rays.append(ray_data[i:i + self.hparams["batch_size"]])

            rendered_batch_rays = self(batchify_rays, 0., 0.)[render_result_key]
            for key in ["rgb_map", "depth_map"]:
                if key not in rendered_rays:
                    rendered_rays[key] = []
                rendered_rays[key].append(rendered_batch_rays[key])

        rendered_rays = {k: torch.concat(rendered_rays[k], 0) for k in rendered_rays}

        gt = extract_rays_rgb(batch["rays"])[0]
        mse = img2mse(rendered_rays["rgb_map"], gt)
        psnr = mse2psnr(mse)

        # reshape image
        H, W = shape[0], shape[1]
        img = rendered_rays["rgb_map"].view(H, W, 3)
        gt_img = gt.view(H, W, 3)

        dssim_ = ssim_loss(img.permute(2, 0, 1)[None,...], gt_img.permute(2, 0, 1)[None,...], 3, reduction="mean")  # dissimilarity in [0, 1]
        ssim = 1 - 2 * dssim_

        # save image
        img = img.cpu()
        gt_img = gt.cpu()
        depth_chw, depth_map = rendering.visualize_depth(rendered_rays[f'depth_map'].view(H, W))

        return {
            "val/loss": mse,
            "val/psnr": psnr,
            "val/ssim": ssim,
            "val/img": img,
            "val/depth_map": depth_map,
            "val/depth_chw": depth_chw,
            "val/gt": gt_img,
        }


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)
