import torch
import numpy as np
import tinycudann as tcnn
from .encodings import Encodings


class TCNNHashGrid(Encodings):
    def __init__(
            self,
            n_input_channels: int,
            n_levels: int,
            n_features_per_level: int,
            log2_hashmap_size: int,
            base_resolution: int,
            desired_resolution: int,
            bounding_box: tuple,
    ):
        super().__init__()

        self.box_min, self.box_max = bounding_box
        self.bound = self.box_max - self.box_min

        # in instant-ngp, max resolution = desired_resolution * bound
        per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (n_levels - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=n_input_channels,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

        self.n_output_channels = self.encoder.n_output_dims

    def forward(self, x):
        box_min = self.box_min.to(x.device)
        box_max = self.box_max.to(x.device)
        bound = self.bound.to(x.device)
        if not torch.all(x <= box_max) or not torch.all(x >= box_min):
            print("ALERT: some points are outside bounding box.")

        x = (x + bound) / (2 * bound)  # normalize to [0., 1.]
        return self.encoder(x)

    def get_output_n_channels(self) -> int:
        return self.n_output_channels
