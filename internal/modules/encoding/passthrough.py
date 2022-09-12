from .encodings import Encodings


class Passthrough(Encodings):
    def __init__(self, input_n_channel: int):
        super().__init__()
        self.input_n_channel = input_n_channel

    def get_output_n_channels(self) -> int:
        return self.input_n_channel
