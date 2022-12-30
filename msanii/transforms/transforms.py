from typing import Any, Dict, Optional, Tuple

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch import Tensor, nn
from torchaudio import transforms as T

from .feature_scaling import MinMaxScaler, StandardScaler
from .griffin_lim import GriffinLim
from .inverse_mel_scale import InverseMelScale


class Transforms(ConfigMixin, nn.Module):
    config_name = "transforms_config.json"

    @register_to_config
    def __init__(
        self,
        sample_rate: int = 44_100,
        n_fft: int = 2048,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        n_mels: int = 128,
        feature_range: Tuple[float, float] = (-1.0, 1.0),
        momentum: float = 1e-3,
        eps: float = 1e-5,
        clip: bool = True,
        num_griffin_lim_iters: int = 50,
        griffin_lim_momentum: float = 0.99,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length or self.n_fft
        self.hop_length = hop_length or (self.win_length // 2)
        self.n_mels = n_mels
        self.num_griffin_lim_iters = num_griffin_lim_iters
        self.griffin_lim_momentum = griffin_lim_momentum

        self.spectrogram = T.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=1.0
        )
        self.complex_spectrogram = T.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
        )
        self.inverse_spectrogram = T.InverseSpectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length
        )
        self.mel_scale = T.MelScale(
            sample_rate=sample_rate, n_mels=n_mels, n_stft=(n_fft // 2 + 1)
        )
        self.inverse_mel_scale = InverseMelScale(
            sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels
        )
        self.griffin_lim = GriffinLim(
            num_iters=num_griffin_lim_iters,
            momentum=griffin_lim_momentum,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=1.0,
        )
        self.standard_scaler = StandardScaler(momentum, eps)
        self.minmax_scaler = MinMaxScaler(feature_range, momentum, clip)

    @property
    def params_dict(self) -> Dict[str, Any]:
        return {
            "standard_scaler/momentum": self.standard_scaler.momentum,
            "standard_scaler/mean": self.standard_scaler.running_mean,
            "standard_scaler/var": self.standard_scaler.running_var,
            "minmax_scaler/momentum": self.minmax_scaler.momentum,
            "minmax_scaler/min": self.minmax_scaler.running_min,
            "minmax_scaler/max": self.minmax_scaler.running_max,
        }

    def forward(
        self, x: Tensor, inverse: bool = False, length: Optional[int] = None
    ) -> Tensor:
        if inverse:
            return self.inverse_transform(x, length)
        return self.transform(x)

    def transform(self, x: Tensor) -> Tensor:
        x_transformed = self.spectrogram(x)
        x_transformed = self.mel_scale(x_transformed)
        x_transformed = torch.log(x_transformed + 1e-5)
        x_transformed = self.standard_scaler(x_transformed)
        x_transformed = self.minmax_scaler(x_transformed)

        return x_transformed

    def inverse_transform(self, x: Tensor, length: Optional[int] = None) -> Tensor:
        x_transformed = self.minmax_scaler(x, inverse=True)
        x_transformed = self.standard_scaler(x_transformed, inverse=True)
        x_transformed = torch.exp(x_transformed)
        x_transformed = self.inverse_mel_scale(x_transformed)
        x_transformed = self.griffin_lim(x_transformed, length=length)

        return x_transformed

    @torch.no_grad()
    def step(self) -> None:
        self.standard_scaler.step()
        self.minmax_scaler.step()
