import torch
import numpy as np
import tinycudann as tcnn
from .encodings import Encodings


class TCNNSphericalHarmonics(Encodings):
    def __init__(
            self,
            n_input_channels: int,
            degree: int,
    ):
        super().__init__()

        self.encoder = tcnn.Encoding(
            n_input_dims=n_input_channels,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
            },
        )

        self.n_output_channels = self.encoder.n_output_dims

    def forward(self, x):
        x = (x + 1) / 2  # normalize to [0., 1.]
        return self.encoder(x)

    def get_output_n_channels(self) -> int:
        return self.n_output_channels
