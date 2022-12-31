import math
import numbers

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from torchaudio import functional as AF


def classname_from_class(classpath) -> str:
    return str(classpath).rsplit(".", maxsplit=1)[1].strip("'>\"")


def compute_divisible_length(
    curr_length: int, hop_length: int, num_downsamples: int
) -> int:
    # Current time frame size
    num_time_frames = int((curr_length / hop_length) + 1)
    # Divisible time frames
    divisible_time_frames = math.ceil(num_time_frames / 2**num_downsamples) * (
        2**num_downsamples
    )
    divisible_length = (divisible_time_frames - 1) * hop_length

    return divisible_length


def pad_to_divisible_length(x: Tensor, hop_length: int, num_downsamples: int) -> Tensor:
    divisible_length = compute_divisible_length(
        x.shape[-1], hop_length, num_downsamples
    )
    # Pad to appropriate length
    x = F.pad(x, (0, divisible_length - x.shape[-1]), value=0.0)

    return x


def gradio_audio_preprocessing(
    audio: np.ndarray,
    src_sample_rate: int,
    target_sample_rate: int,
    target_length: int,
    hop_length: int,
    num_downsamples: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    # Ensure audio is a float tensor between [-1, 1]
    if issubclass(audio.dtype.type, numbers.Integral):
        audio = audio / np.iinfo(audio.dtype).max

    # Load audio into tensor and resample
    audio = torch.from_numpy(audio)
    if audio.ndim == 1:
        audio = rearrange(audio, "l -> () () l")  # to batched and mono-channels
    else:
        audio = rearrange(audio, "l c -> () c l")  # to batched channel first
    audio = AF.resample(audio, src_sample_rate, target_sample_rate)

    # Pad audio to the target length
    audio = F.pad(audio, (0, target_length - audio.shape[-1]))

    # Pad audio to a length divisible by the number of downsampling layers
    audio = pad_to_divisible_length(audio, hop_length, num_downsamples)

    # Switch target dtype and device
    audio = audio.to(dtype).to(device)

    return audio


def gradio_audio_postprocessing(audio: Tensor, target_length: int) -> np.ndarray:
    # Ensure audio is the correct length
    audio = F.pad(audio, (0, target_length - audio.shape[-1]))

    # Remove batch dimension & switch to channels last
    audio = rearrange(audio, "b c l -> (l b) c")

    return audio.detach().cpu().numpy()
