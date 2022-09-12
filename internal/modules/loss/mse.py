import torch
from torch import nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.img2mse = lambda x, y: torch.mean((x - y) ** 2)
        self.mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

    def forward(self, inputs, targets):
        coarse_loss = self.img2mse(inputs['coarse']['rgb_map'], targets)
        fine_loss = self.img2mse(inputs['fine']['rgb_map'], targets)

        with torch.no_grad():
            coarse_psnr = self.mse2psnr(coarse_loss)
            fine_psnr = self.mse2psnr(fine_loss)

        return coarse_loss, fine_loss, coarse_psnr, fine_psnr
