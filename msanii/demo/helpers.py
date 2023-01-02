import numbers
from typing import Tuple

import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torch.nn import functional as F
from torchaudio import functional as AF

from ..pipeline import Pipeline
from ..utils import plot_spectrogram, plot_waveform
from .utils import (
    compute_divisible_length,
    generate_gradio_audio_mask,
    gradio_audio_postprocessing,
    gradio_audio_preprocessing,
)


def run_sampling(
    pipeline: Pipeline,
    duration: int,
    channels: int,
    num_inference_steps: int,
    eta: float,
    use_neural_vocoder: bool,
    num_griffin_lim_iters: int,
    seed: float,
) -> Tuple[Figure, Figure, Tuple[int, np.ndarray]]:
    # Prepare sample audio that will guide the sampling process
    audio_length = duration * pipeline.transforms.sample_rate
    divisible_length = compute_divisible_length(
        audio_length, pipeline.transforms.hop_length, sum(pipeline.unet.has_resampling)
    )
    audio = torch.randn(
        (1, channels, divisible_length), dtype=pipeline.dtype, device=pipeline.device
    )

    # Generate sample from the pipeline
    generator = torch.Generator(pipeline.device).manual_seed(int(seed))
    audio = pipeline.sample(
        audio,
        num_inference_steps,
        generator=generator,
        use_neural_vocoder=use_neural_vocoder,
        num_griffin_lim_iters=num_griffin_lim_iters,
        eta=eta,
    )

    # Compute waveform and spectrogram representation
    spectrogram = plot_spectrogram(pipeline.transforms(audio))
    waveform = plot_waveform(audio, pipeline.transforms.sample_rate)
    audio = gradio_audio_postprocessing(audio, audio_length)

    return spectrogram, waveform, (pipeline.transforms.sample_rate, audio)


def run_audio2audio(
    pipeline: Pipeline,
    audio: Tuple[int, np.ndarray],
    duration: int,
    num_inference_steps: int,
    strength: float,
    use_neural_vocoder: bool,
    num_griffin_lim_iters: int,
    seed: float,
    eta: float,
) -> Tuple[Figure, Figure, Tuple[int, np.ndarray]]:
    # Convert audio to tensor & resample
    sample_rate, audio = audio

    # Apply some preprocessing
    target_length = duration * pipeline.transforms.sample_rate
    audio = gradio_audio_preprocessing(
        audio,
        src_sample_rate=sample_rate,
        target_sample_rate=pipeline.transforms.sample_rate,
        target_length=target_length,
        hop_length=pipeline.transforms.hop_length,
        num_downsamples=sum(pipeline.unet.has_resampling),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )

    # Generate sample from pipeline
    generator = torch.Generator(pipeline.device).manual_seed(int(seed))
    audio = pipeline.sample(
        audio,
        num_inference_steps,
        strength=strength,
        generator=generator,
        use_neural_vocoder=use_neural_vocoder,
        use_input_as_seed=True,
        num_griffin_lim_iters=num_griffin_lim_iters,
        eta=eta,
    )

    # Compute waveform and spectrogram representation
    spectrogram = plot_spectrogram(pipeline.transforms(audio))
    waveform = plot_waveform(audio, pipeline.transforms.sample_rate)
    audio = gradio_audio_postprocessing(audio, target_length)

    return spectrogram, waveform, (pipeline.transforms.sample_rate, audio)


def run_interpolation(
    pipeline: Pipeline,
    first_audio: Tuple[int, np.ndarray],
    second_audio: Tuple[int, np.ndarray],
    duration: int,
    num_inference_steps: int,
    ratio: float,
    strength: float,
    use_neural_vocoder: bool,
    num_griffin_lim_iters: int,
    seed: float,
    eta: float,
) -> Tuple[Figure, Figure, Tuple[int, np.ndarray]]:
    # Convert audio to tensor & resample
    first_sample_rate, first_audio = first_audio
    second_sample_rate, second_audio = second_audio

    # Apply some preprocessing
    target_length = duration * pipeline.transforms.sample_rate
    first_audio = gradio_audio_preprocessing(
        first_audio,
        src_sample_rate=first_sample_rate,
        target_sample_rate=pipeline.transforms.sample_rate,
        target_length=target_length,
        hop_length=pipeline.transforms.hop_length,
        num_downsamples=sum(pipeline.unet.has_resampling),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )
    second_audio = gradio_audio_preprocessing(
        second_audio,
        src_sample_rate=second_sample_rate,
        target_sample_rate=pipeline.transforms.sample_rate,
        target_length=target_length,
        hop_length=pipeline.transforms.hop_length,
        num_downsamples=sum(pipeline.unet.has_resampling),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )

    # Generate sample from pipeline
    generator = torch.Generator(pipeline.device).manual_seed(int(seed))
    audio = pipeline.interpolate(
        first_audio,
        second_audio,
        ratio=ratio,
        num_inference_steps=num_inference_steps,
        strength=strength,
        generator=generator,
        use_neural_vocoder=use_neural_vocoder,
        num_griffin_lim_iters=num_griffin_lim_iters,
        eta=eta,
    )

    # Compute waveform and spectrogram representation
    spectrogram = plot_spectrogram(pipeline.transforms(audio))
    waveform = plot_waveform(audio, pipeline.transforms.sample_rate)
    audio = gradio_audio_postprocessing(audio, target_length)

    return spectrogram, waveform, (pipeline.transforms.sample_rate, audio)


def run_inpainting(
    pipeline: Pipeline,
    audio: Tuple[int, np.ndarray],
    mask_spec: str,
    jump_length: int,
    jump_n_samples: int,
    duration: int,
    num_inference_steps: int,
    use_neural_vocoder: bool,
    num_griffin_lim_iters: int,
    seed: float,
    eta: float,
) -> Tuple[Figure, Figure, Tuple[int, np.ndarray]]:
    sample_rate, audio = audio

    # Generate mask from the mask-spec
    audio_mask = generate_gradio_audio_mask(audio, sample_rate, mask_spec)

    print(audio.shape, audio_mask.shape)

    # Apply some preprocessing
    target_length = duration * pipeline.transforms.sample_rate
    audio = gradio_audio_preprocessing(
        audio,
        src_sample_rate=sample_rate,
        target_sample_rate=pipeline.transforms.sample_rate,
        target_length=target_length,
        hop_length=pipeline.transforms.hop_length,
        num_downsamples=sum(pipeline.unet.has_resampling),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )
    audio_mask = gradio_audio_preprocessing(
        audio_mask.astype(float),
        src_sample_rate=sample_rate,
        target_sample_rate=pipeline.transforms.sample_rate,
        target_length=target_length,
        hop_length=pipeline.transforms.hop_length,
        num_downsamples=sum(pipeline.unet.has_resampling),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )

    # Generate sample from pipeline
    generator = torch.Generator(pipeline.device).manual_seed(int(seed))
    audio = pipeline.inpaint(
        audio,
        audio_mask,
        num_inference_steps=num_inference_steps,
        jump_length=jump_length,
        jump_n_sample=jump_n_samples,
        generator=generator,
        use_neural_vocoder=use_neural_vocoder,
        num_griffin_lim_iters=num_griffin_lim_iters,
        eta=eta,
    )

    # Compute waveform and spectrogram representation
    spectrogram = plot_spectrogram(pipeline.transforms(audio))
    waveform = plot_waveform(audio, pipeline.transforms.sample_rate)
    audio = gradio_audio_postprocessing(audio, target_length)

    return spectrogram, waveform, (pipeline.transforms.sample_rate, audio)


def run_outpainting(
    pipeline: Pipeline,
    audio: Tuple[int, np.ndarray],
    num_spans: int,
    duration: int,
    num_inference_steps: int,
    use_neural_vocoder: bool,
    num_griffin_lim_iters: int,
    seed: float,
    eta: float,
) -> Tuple[Figure, Figure, Tuple[int, np.ndarray]]:
    # Convert audio to tensor & resample
    sample_rate, audio = audio

    # Apply some preprocessing
    seed_length = duration * pipeline.transforms.sample_rate
    audio = gradio_audio_preprocessing(
        audio,
        src_sample_rate=sample_rate,
        target_sample_rate=pipeline.transforms.sample_rate,
        target_length=seed_length,
        hop_length=pipeline.transforms.hop_length,
        num_downsamples=sum(pipeline.unet.has_resampling),
        dtype=pipeline.dtype,
        device=pipeline.device,
    )

    # Generate sample from pipeline
    generator = torch.Generator(pipeline.device).manual_seed(int(seed))
    audio = pipeline.outpaint(
        audio,
        num_spans=num_spans,
        num_inference_steps=num_inference_steps,
        generator=generator,
        use_neural_vocoder=use_neural_vocoder,
        num_griffin_lim_iters=num_griffin_lim_iters,
        eta=eta,
    )

    # Compute waveform and spectrogram representation
    target_length = int(seed_length + (((duration / 2) * num_spans) * sample_rate))
    spectrogram = plot_spectrogram(pipeline.transforms(audio))
    waveform = plot_waveform(audio, pipeline.transforms.sample_rate)
    audio = gradio_audio_postprocessing(audio, target_length)

    return spectrogram, waveform, (pipeline.transforms.sample_rate, audio)
