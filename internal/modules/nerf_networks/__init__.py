import torch.nn
from internal.modules.nerf_networks import nerf, tcnn_cutlass, tcnn_ff


def get_nerf_network(
        location_input_channels: int,
        view_direction_input_channels: int,
        hparams: dict
) -> torch.nn.Module:
    network_parameters = {
        "density_layers": hparams["density_layers"],
        "density_layer_units": hparams["density_layer_units"],
        "color_layers": hparams["color_layers"],
        "color_layer_units": hparams["color_layer_units"],
        "location_input_channels": location_input_channels,
        "view_direction_input_channels": view_direction_input_channels,
    }

    # set default network type
    if "network_type" not in hparams:
        network_type = "paper"
    else:
        network_type = hparams["network_type"]

    # paper network
    if network_type == "paper":
        # network parameters
        network_parameters.update({
            "skips": hparams["skips"],
        })
        network_class = nerf.NeRF
    elif hparams["network_type"] == "tcnn_ff":
        network_parameters.update({
            "density_output_features": hparams["density_output_features"],
        })
        network_class = tcnn_ff.TCNNFullyFusedNeRF
    elif hparams["network_type"] == "tcnn_cutlass":
        network_parameters.update({
            "skips": hparams["skips"],
        })
        network_class = tcnn_cutlass.TCNNCutlass
    else:
        raise ValueError(f"unsupported network type: {network_type}")

    # create network instance
    return network_class(**network_parameters)
