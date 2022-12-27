import torch
from torch import Tensor, nn

from ..modules import InputFn, OutputFn, ResidualBlock


class Vocoder(nn.Module):
    def __init__(
        self,
        n_fft: int = 2048,
        n_mels: int = 128,
        d_model: int = 256,
        d_hidden_factor: int = 4,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.d_model = d_model
        self.d_hidden_factor = d_hidden_factor

        self.fn = nn.Sequential(
            InputFn(self.n_mels, self.d_model),
            ResidualBlock(self.d_model, self.d_model, self.d_hidden_factor),
            OutputFn((self.n_fft // 2 + 1), self.d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(self.fn(x))
