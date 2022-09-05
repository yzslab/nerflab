import torch
from torch import nn
import pytorch_lightning as pl


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, input_channels, n_frequencies, log_sampling=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.n_frequencies = n_frequencies
        self.input_channels = input_channels
        self.funcs = [torch.sin, torch.cos]
        self.output_channels = input_channels * (len(self.funcs) * n_frequencies + 1)

        max_frequencies = n_frequencies - 1
        if log_sampling:
            self.freq_bands = 2.**torch.linspace(0., max_frequencies, steps=n_frequencies)

        else:
            self.freq_bands = torch.linspace(2.**0., 2.**max_frequencies, steps=n_frequencies)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


# NeRF network
class NeRF(nn.Module):
    def __init__(
            self,
            density_layers=8,
            density_layer_units=256,
            color_layers=1,
            color_layer_units=128,
            skips=[4],
            location_encoder=None,
            location_input_channels=63,  # 3+3*10*2=63 by default;
            view_direction_encoder=None,
            view_direction_input_channels=27,  # 3+3*4*2=27 by default
    ):
        """
        :param density_layers:
        :param density_layer_units:
        :param color_layers:
        :param color_layer_units:
        :param skips:
        :param location_encoder:
        :param location_input_channels:
            3D location: (x, y, z),
            each value of axis encoded by 10 (sin, cos),
            so there is a vector with length of 3x2x10=60,
            then concat with original (x, y, z)
        :param view_direction_encoder:
        :param view_direction_input_channels:
        """

        super(NeRF, self).__init__()

        # store parameters
        self.density_layers = density_layers
        self.density_layer_units = density_layer_units
        self.color_layers = color_layers
        self.color_layer_units = color_layer_units
        self.skips = skips
        self.location_encoder = location_encoder
        self.location_input_channels = location_input_channels
        self.view_direction_encoder = view_direction_encoder
        self.view_direction_input_channels = view_direction_input_channels

        # density network ReLU layers (8 full-connected ReLU layers)
        self.density_relu_layers = []

        modules = [nn.Linear(location_input_channels, density_layer_units)]
        for i in range(density_layers - 1):
            if i in skips:
                # create nn.Sequential() before skip connection
                self.density_relu_layers.append(nn.Sequential(*modules))
                modules = [nn.Linear(density_layer_units + location_input_channels, density_layer_units), nn.ReLU()]
            else:
                modules.append(nn.Linear(density_layer_units, density_layer_units))
                modules.append(nn.ReLU())
        if len(modules) > 0:
            self.density_relu_layers.append(nn.Sequential(*modules))

        # density network output layers
        # one for output sigma
        # another for output feature vector as the input for the RGB network
        self.density_output_layers = [
            nn.Linear(density_layer_units, 1),  # output sigma
            nn.Linear(density_layer_units, density_layer_units),  # output 256-dimensions vector as RGB layer input
        ]

        # RGB network
        self.rgb_network = nn.Sequential(
            nn.Linear(view_direction_input_channels, color_layer_units),
            nn.ReLU(),
            nn.Linear(color_layer_units, 3),  # output layer
            nn.Sigmoid(),
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

        # pass through ReLU layers
        density_output = []
        for module in self.density_relu_layers:
            density_output = torch.cat([encoded_location, density_output], -1)  # skip connection
            density_output = module(density_output)

        sigma = self.density_output_layers[0](density_output)

        rgb_network_input_feature = torch.cat([
            self.density_output_layers[1](density_output),
            encoded_view_direction,
        ], -1)
        rgb_network_output = self.rgb_network(rgb_network_input_feature)

        return torch.cat([rgb_network_output, sigma], -1)
