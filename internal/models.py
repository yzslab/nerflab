import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import SGD, Adam
from internal.modules import PositionalEncoding
from internal.modules import NeRF as NeRFNetwork

class NeRF(pl.LightningModule):
    def __init__(
            self,
            hparams,
    ):
        super().__init__()
        self.save_hyperparameters(hparams),

        self.location_encoder = PositionalEncoding(3, 10)
        self.view_direction_encoder = PositionalEncoding(3, 4)

        # coarse, N_sample
        self.coarse_network = NeRFNetwork(location_encoder=self.location_encoder,
                                          view_direction_encoder=self.view_direction_encoder)
        # fine, N_importance
        self.fine_network = NeRFNetwork(location_encoder=self.location_encoder,
                                        view_direction_encoder=self.view_direction_encoder)

    def forward(self, rays):
        pass

    def configure_optimizers(self):
        lr=5e-4
        eps = 1e-8
        parameters = []

        parameters += list(self.coarse_network.parameters())
        parameters += list(self.fine_network.parameters())

        self.optimizer = Adam(parameters, lr=lr)
