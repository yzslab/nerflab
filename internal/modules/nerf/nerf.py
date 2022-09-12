import torch
from torch import nn


# NeRF network
class NeRF(nn.Module):
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
        super(NeRF, self).__init__()

        # store parameters
        self.density_layers = density_layers
        self.density_layer_units = density_layer_units
        self.color_layers = color_layers
        self.color_layer_units = color_layer_units
        self.skips = skips
        self.location_input_channels = location_input_channels
        self.view_direction_input_channels = view_direction_input_channels

        # density network ReLU layers (8 full-connected ReLU layers)
        self.density_relu_layers = nn.ModuleList([])

        modules = nn.ModuleList([nn.Linear(location_input_channels, density_layer_units)])
        for i in range(density_layers - 1):
            if i in skips:
                # create nn.Sequential() before skip connection
                self.density_relu_layers.append(nn.Sequential(*modules))
                modules = nn.ModuleList(
                    [nn.Linear(density_layer_units + location_input_channels, density_layer_units), nn.ReLU()])
            else:
                modules.append(nn.Linear(density_layer_units, density_layer_units))
                modules.append(nn.ReLU())
        if len(modules) > 0:
            self.density_relu_layers.append(nn.Sequential(*modules))

        # density network output layers
        # one for output sigma
        # another for output feature vector as the input for the RGB network
        self.density_output_layers = nn.ModuleList([
            nn.Linear(density_layer_units, 1),  # output sigma
            nn.Linear(density_layer_units, density_layer_units),  # output 256-dimensions vector as RGB layer input
        ])

        # RGB network
        self.rgb_network = nn.Sequential(
            # input: density layers output feature + encoded view direction
            nn.Linear(view_direction_input_channels + density_layer_units, color_layer_units),
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
        density_output = torch.tensor([], device=x.device)
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
