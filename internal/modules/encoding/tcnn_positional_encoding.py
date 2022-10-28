import torch
import tinycudann as tcnn
from .encodings import Encodings


class TCNNPositionalEncoding(Encodings):
    def __init__(self, input_channels: int, n_frequencies: int):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.input_channels = input_channels

        self.encoder = tcnn.Encoding(
            n_input_dims=input_channels,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": n_frequencies
            },
        )

        self.n_output_channels = self.encoder.n_output_dims

    def forward(self, x):
        return self.encoder(x)

    def get_output_n_channels(self) -> int:
        return self.n_output_channels
