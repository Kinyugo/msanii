import torch
from einops import reduce
from matplotlib import pyplot as plt
from torch import Tensor, nn


def plot_waveform(waveform: Tensor, sample_rate: int, title: str = "") -> plt.Figure:
    waveform = reduce(waveform, "... l -> l", reduction="mean")
    waveform = waveform.detach().cpu()

    n_frames = waveform.shape[-1]
    skip = int(n_frames / (0.01 * n_frames))
    waveform = waveform[..., 0:-1:skip]

    n_frames = waveform.shape[-1]
    time_axis = torch.linspace(0, n_frames / (sample_rate / skip), steps=n_frames)

    fig = plt.figure(dpi=1200)
    plt.plot(time_axis, waveform, linewidth=1)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    return fig


def plot_spectrogram(spectrogram: Tensor, title: str = "") -> plt.Figure:
    spectrogram = reduce(spectrogram, "... f t-> f t", reduction="mean")
    spectrogram = spectrogram.detach().cpu()

    fig = plt.figure(dpi=1200)
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    return fig


def plot_distribution(x: Tensor, title: str = "") -> plt.Figure:
    x = x.detach().cpu()
    mean, std = x.mean(), x.std()

    hist, edges = torch.histogram(x, density=True)

    fig = plt.figure(dpi=1200)
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
