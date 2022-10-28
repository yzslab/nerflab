import torch
from torch import nn
import tinycudann as tcnn


class TCNNCutlass(nn.Module):
    def __init__(
            self,
            density_layers: int,
            density_layer_units: int,
            color_layers: int,
            color_layer_units: int,
            skips: list,
            location_input_channels: int,  # 3+3*10*2=63 by default;
            view_direction_input_channels: int,  # 3+3*4*2=27 by default
    ):
        super().__init__()

        # store parameters
        self.density_layers = density_layers
        self.density_layer_units = density_layer_units
        self.color_layers = color_layers
        self.color_layer_units = color_layer_units
        self.skips = skips
        self.location_input_channels = location_input_channels
        self.view_direction_input_channels = view_direction_input_channels

        # density network ReLU layers (8 full-connected ReLU layers)
        self.density_network = tcnn.Network(
            n_input_dims=location_input_channels,
            n_output_dims=1 + density_layer_units,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": density_layer_units,
                "n_hidden_layers": density_layers,
            }
        )

        # rgb network
        self.rgb_network = tcnn.Network(
            n_input_dims=view_direction_input_channels + density_layer_units,
            n_output_dims=3,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": color_layer_units,
                "n_hidden_layers": color_layers,
            }
        )

    def forward(self, x):
        """
        :param x: [0-62: sample point location PE, 63-89: view direction PE]
        :return: [RGB, sigma]
        """

        encoded_location, encoded_view_direction = torch.split(x,
                                                               [self.location_input_channels,
                                                                self.view_direction_input_channels],
                                                               dim=-1)

        density_network_output = self.density_network(encoded_location)
        sigma = density_network_output[..., 0:1]
        feature_vector = density_network_output[..., 1:]

        rgb_network_input_feature = torch.cat([
            feature_vector,
            encoded_view_direction,
        ], -1)
        rgb_network_output = self.rgb_network(rgb_network_input_feature)

        return torch.cat([rgb_network_output, sigma], -1)
