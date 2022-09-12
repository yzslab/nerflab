from typing import Tuple
from .encodings import Encodings
from .positional_encoding import PositionalEncoding
from .passthrough import Passthrough


def get_encoding(hparams: dict) -> Tuple[Encodings, Encodings]:
    name = hparams["encoding"]

    if name == "pe":
        return PositionalEncoding(input_channels=3, n_frequencies=hparams["pe_location_n_freq"]), \
               PositionalEncoding(input_channels=3, n_frequencies=hparams["pe_direction_n_freq"])
    if name == "passthrough":
        return Passthrough(3), Passthrough(3)

    raise ValueError(f"unsupported encoding type: {name}")
