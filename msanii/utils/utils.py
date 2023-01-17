import math

import torch
from einops import reduce
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F


def plot_waveform(waveform: Tensor, sample_rate: int, title: str = "") -> plt.Figure:
    waveform = reduce(waveform, "... l -> l", reduction="mean")
    waveform = waveform.detach().cpu()

    n_frames = waveform.shape[-1]
    skip = int(n_frames / (0.01 * n_frames))
    waveform = waveform[..., 0:-1:skip]

    n_frames = waveform.shape[-1]
    time_axis = torch.linspace(0, n_frames / (sample_rate / skip), steps=n_frames)

    fig = plt.figure(dpi=300)
    plt.plot(time_axis, waveform, linewidth=1)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    return fig


def plot_spectrogram(spectrogram: Tensor, title: str = "") -> plt.Figure:
    spectrogram = reduce(spectrogram, "... f t-> f t", reduction="mean")
    spectrogram = spectrogram.detach().cpu()

    fig = plt.figure(dpi=300)
    plt.imshow(spectrogram, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    return fig


def plot_distribution(x: Tensor, title: str = "") -> plt.Figure:
    x = x.detach().cpu()
    mean, std = x.mean(), x.std()

    hist, edges = torch.histogram(x, density=True)

    fig = plt.figure(dpi=300)
    plt.plot(edges[:-1], hist)
    plt.title(f"{title} | Mean: {mean:.4f} Std: {std:.4f}")
    plt.xlabel("X")
    plt.ylabel("Density")

    return fig


def freeze_model(model: nn.Module) -> nn.Module:
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def clone_model_parameters(src_model: nn.Module, target_model: nn.Module) -> nn.Module:
    for src_param, target_param in zip(
        src_model.parameters(), target_model.parameters()
    ):
        target_param.data = src_param.data

    return target_model


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


def pad_to_divisible_length(
    x: Tensor, hop_length: int, num_downsamples: int, pad_end: bool = True
) -> Tensor:
    divisible_length = compute_divisible_length(
        x.shape[-1], hop_length, num_downsamples
    )
    # Pad to appropriate length
    if pad_end:
        x = F.pad(x, (0, divisible_length - x.shape[-1]))
    else:
        x = F.pad(x, (divisible_length - x.shape[-1], 0))

    return x
