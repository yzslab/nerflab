import torch
from internal.modules.encoding.encodings import Encodings


class IntegratedPositionalEncoding(Encodings):
    def __init__(self, input_channels: int, n_frequencies=10):
        super(IntegratedPositionalEncoding, self).__init__()
        self.n_frequencies = n_frequencies
        self.funcs = [torch.sin, torch.cos]

        self.output_channels = input_channels * len(self.funcs) * n_frequencies

        # [2^0,2^1,...,2^(n-1)]: for sin
        self.freq_band_1 = 2 ** torch.linspace(0, n_frequencies - 1, n_frequencies)
        # [4^0,4^1,...,4^(n-1)]: for diag(∑)
        self.freq_band_2 = self.freq_band_1 ** 2

    def forward(self, mu, diagE):
        # exmbeds [mu,diagE] -> [sin(mu)*exp(-1/2)*diag(∑gamma),cos(mu)*exp(-1/2)*diag(∑gamma),...,
        # sin(2^(L-1)*mu)*exp(-1/2)*4^(L-1)*diag(∑)]
        sin_out = []
        sin_cos = []
        for freq in self.freq_band_1:
            for func in self.funcs:
                sin_cos.append(func(freq * mu))
            sin_out.append(sin_cos)
            sin_cos = []
        # sin_out:list:[sin(mu),cos(mu)]
        diag_out = []
        for freq in self.freq_band_2:
            diag_out.append(freq * diagE)
        # diag_out:list:[4^(L-1)*diag(∑)]
        out = []
        for sc_gamma, diag_expectation_gamma in zip(sin_out, diag_out):
            # torch.exp(-0.5 * x_var) * torch.sin(x)
            for sin_cos in sc_gamma:  # [sin,cos]
                out.append(sin_cos * torch.exp(-0.5 * diag_expectation_gamma))
        return torch.cat(out, -1)

    def get_output_n_channels(self) -> int:
        return self.output_channels
