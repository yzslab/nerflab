import torch
from torch import nn

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        coarse_loss = img2mse(inputs['coarse']['rgb_map'], targets)
        fine_loss = -1
        if 'fine' in inputs:
            fine_loss = img2mse(inputs['fine']['rgb_map'], targets)

        with torch.no_grad():
            coarse_psnr = mse2psnr(coarse_loss)
            fine_psnr = -1
            if 'fine' in inputs:
                fine_psnr = mse2psnr(fine_loss)

        return coarse_loss, fine_loss, coarse_psnr, fine_psnr
