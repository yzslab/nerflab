import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import tinycudann as tcnn

from torch import nn


# tiny-cuda-nn fully fused MLP
class TCNNFullyFusedNeRF(nn.Module):
    def __init__(
            self,
            density_layers: int,
            density_layer_units: int,
            color_layers: int,
            color_layer_units: int,
            location_input_channels: int,
            view_direction_input_channels: int,
            density_output_features: int = 15
    ):
        super().__init__()

        self.location_input_channels = location_input_channels
        self.view_direction_input_channels = view_direction_input_channels

        # [0: sigma value, 1-15: feature vector as rgb network input]
        density_output_dims = density_output_features + 1

        # create density network
        self.density_network = tcnn.Network(
            n_input_dims=location_input_channels,
            n_output_dims=density_output_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": density_layer_units,
                "n_hidden_layers": density_layers - 1,
            },
        )

        # create rgb network
        self.rgb_network = tcnn.Network(
            n_input_dims=view_direction_input_channels + density_output_features,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": color_layer_units,
                "n_hidden_layers": color_layers - 1,
            },
        )

    def forward(self, x):
        # extract encoded location and view direction
        encoded_location, encoded_view_direction = torch.split(x,
                                                               [self.location_input_channels,
                                                                self.view_direction_input_channels],
                                                               dim=-1)

        # query density network
        density_output = self.density_network(encoded_location)
        # get sigma without activation function
        sigma = density_output[..., 0:1]
        density_output_feature_vector = density_output[..., 1:]

        rgb_network_input_feature_vector = torch.cat([encoded_view_direction, density_output_feature_vector], dim=-1)
        rgb_network_output = self.rgb_network(rgb_network_input_feature_vector)

        rgb = torch.sigmoid(rgb_network_output)

        return torch.cat([rgb, sigma], -1)


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _trunc_exp.apply
