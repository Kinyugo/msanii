from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import Tensor, nn

from .utils import get_complex_dtype


class BaseIterativeVocoder(ABC, nn.Module):
    def __init__(
        self,
        num_iters: int = 100,
        n_fft: int = 2048,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        wkwargs: Optional[dict] = None,
        power: float = 1.0,
        eps: float = 1e-16,
    ) -> None:
        super().__init__()

        self.num_iters = num_iters
        self.n_fft = n_fft
        self.win_length = n_fft if win_length is None else win_length
        self.hop_length = self.win_length // 2 if hop_length is None else hop_length
        window = (
            window_fn(self.win_length)
            if wkwargs is None
            else window_fn(self.win_length, **wkwargs)
        )
        self.register_buffer("window", window)
        self.power = power
        self.eps = eps

    def forward(
        self,
        specgram: Tensor,
        *,
        init_phase: Optional[Tensor] = None,
        length: Optional[int] = None,
        return_phase: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # Flatten the specgram except the frequency and time dimension
        shape = specgram.shape
        specgram = rearrange(specgram, "... f t -> (...) f t")

        # Project arbitrary specgram into a magnitude specgram
        specgram = specgram.pow(1 / self.power)

        # Initialize the phase
        if init_phase is None:
            phase = torch.rand(
                specgram.shape,
                dtype=get_complex_dtype(specgram.dtype),
                device=specgram.device,
            )
        else:
            phase = init_phase / (torch.abs(init_phase) + self.eps)
            phase = phase.reshape(specgram.shape)

        # Reconstruct the phase
        phase = self.reconstruct_phase(specgram, phase=phase, length=length)

        # Synthesize the waveform using computed phase
        waveform = torch.istft(
            specgram * phase,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window,
            length=length,
        )

        # Unflatten the waveform & phase
        waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])
        phase = phase.reshape(shape)

        if return_phase:
            return waveform, phase
        return waveform

    @abstractmethod
    def reconstruct_phase(
        self,
        specgram: Tensor,
        *,
        phase: Optional[Tensor] = None,
        length: Optional[int] = None,
    ) -> Tensor:
        raise NotImplementedError

    def project_onto_magspec(self, magspec: Tensor, stftspec: Tensor) -> Tensor:
        return magspec * stftspec / (torch.abs(stftspec) + self.eps)

    def project_complex_spec(
        self, stftspec: Tensor, length: Optional[int] = None
    ) -> Tensor:
        # Invert with our current phase estimates
        inverse = torch.istft(
            stftspec,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window,
            length=length,
        )
        # inverse = inverse.clamp(min=-1.0, max=1.0)

        # Rebuild the complex spectrogram
        rebuilt = torch.stft(
            inverse,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        return rebuilt


class GriffinLim(BaseIterativeVocoder):
    def __init__(
        self,
        num_iters: int = 100,
        momentum: float = 0.99,
        n_fft: int = 2048,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        wkwargs: Optional[dict] = None,
        power: float = 1.0,
        eps: float = 1e-16,
    ) -> None:
        super().__init__(
            num_iters, n_fft, win_length, hop_length, window_fn, wkwargs, power, eps
        )

        self.momentum = momentum

    def reconstruct_phase(
        self,
        specgram: Tensor,
        *,
        phase: Optional[Tensor] = None,
        length: Optional[int] = None,
    ) -> Tensor:
        momentum = self.momentum / (1 + self.momentum)

        # Initialize our previous iterate
        prev_stftspec = torch.tensor(0.0, dtype=specgram.dtype, device=specgram.device)

        for _ in range(self.num_iters):
            # Invert and rebuild with the current phase estimate
            next_stftspec = self.project_complex_spec(specgram * phase, length)

            # Update our phase estimates
            phase = next_stftspec - (prev_stftspec * momentum)
            phase = phase / (torch.abs(phase) + self.eps)

            # Update our previous iterate
            prev_stftspec = next_stftspec

        return phase
