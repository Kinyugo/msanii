from typing import Optional

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torchaudio import functional as F


class InverseMelScale(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> None:
        super().__init__()

        # Compute the inverse filter banks using the pseudo inverse
        f_max = f_max or float(sample_rate // 2)
        fb = F.melscale_fbanks(
            (n_fft // 2 + 1), f_min, f_max, n_mels, sample_rate, norm, mel_scale
        )
        # Using pseudo-inverse is faster than calculating the least-squares in each
        # forward pass and experiments show that they converge to the same solution
        self.register_buffer("fb", torch.linalg.pinv(fb))

    def forward(self, melspec: Tensor) -> Tensor:
        # Flatten the melspec except for the frequency and time dimension
        shape = melspec.shape
        melspec = rearrange(melspec, "... f t -> (...) f t")

        # Expand the filter banks to match the melspec
        fb = repeat(self.fb, "f m -> n m f", n=melspec.shape[0])

        # Sythesize the stft specgram using the filter banks
        specgram = fb @ melspec
        # Ensure non-negative solution
        specgram = torch.clamp(specgram, min=0.0)

        # Unflatten the specgram (*, freq, time)
        specgram = specgram.view(shape[:-2] + (fb.shape[-2], shape[-1]))

        return specgram
