from typing import Tuple
from .encodings import Encodings
from .positional_encoding import PositionalEncoding
from .passthrough import Passthrough
from .tcnn_hash_grid import TCNNHashGrid
from .tcnn_spherical_harmonics import TCNNSphericalHarmonics


def get_encoding(hparams: dict) -> Tuple[Encodings, Encodings]:
    location_encoding_name = hparams["location_encoding"]
    view_direction_encoding_name = hparams["viewdir_encoding"]

    if location_encoding_name == "pe":
        location_encoding = PositionalEncoding(input_channels=3, n_frequencies=hparams["pe_location_n_freq"])
    elif location_encoding_name == "passthrough":
        location_encoding = Passthrough(3)
    elif location_encoding_name == "tcnn_hash_grid":
        location_encoding = TCNNHashGrid(
            bounding_box=hparams["bounding_box"],
            **hparams["hash_grid"],
        )
    else:
        raise ValueError(f"unsupported location encoding: {location_encoding_name}")

    if view_direction_encoding_name == "pe":
        view_direction_encoding = PositionalEncoding(input_channels=3, n_frequencies=hparams["pe_direction_n_freq"])
    elif view_direction_encoding_name == "passthrough":
        view_direction_encoding = Passthrough(3)
    elif view_direction_encoding_name == "tcnn_sh":
        view_direction_encoding = TCNNSphericalHarmonics(**hparams["spherical_harmonics"])
    else:
        raise ValueError(f"unsupported view direction encoding: {view_direction_encoding_name}")

    return location_encoding, view_direction_encoding
