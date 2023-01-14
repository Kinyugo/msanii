import math
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from torch import Tensor
from tqdm.autonotebook import tqdm

from ..config import (
    Audio2AudioConfig,
    InpaintingConfig,
    InterpolationConfig,
    OutpaintingConfig,
    SamplingConfig,
)
from ..data import AudioDataModule
from ..pipeline import Pipeline
from ..utils import compute_divisible_length
from .utils import generate_batch_audio_mask


def save_batch_samples(
    samples: Tensor, offset: int, output_dir: str, sample_rate: int, audio_format: str
) -> None:
    # torchaudio can only save cpu samples
    samples = samples.detach().cpu()

    os.makedirs(output_dir, exist_ok=True)
    for i, sample in enumerate(samples):
        filename = f"{offset + i}.{audio_format}"
        filepath = os.path.join(output_dir, filename)
        torchaudio.save(filepath, sample, sample_rate=sample_rate, format=audio_format)


def run_sampling(config: SamplingConfig) -> None:
    # -------------------------------------------
    # Setup
    # -------------------------------------------
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)

    # -------------------------------------------
    # Load pipeline
    # -------------------------------------------
    pipeline = Pipeline.from_pretrained(config.ckpt_path).to(device).to(dtype)

    # -------------------------------------------
    # Run sampling
    # -------------------------------------------
    n_batches = config.num_samples // config.batch_size
    batches = np.array_split(range(config.num_samples), n_batches)

    # Optionally compute the number of frames from duration
    num_frames = config.num_frames
    if config.duration is not None:
        num_frames = config.duration * pipeline.transforms.sample_rate
        num_frames = compute_divisible_length(
            num_frames,
            pipeline.transforms.hop_length,
            sum(pipeline.unet.has_resampling),
        )

    for batch_idx, batch in enumerate(
        tqdm(batches, desc="Sampling", disable=(not config.verbose))
    ):
        samples = torch.randn(
            (len(batch), config.channels, num_frames),
            device=pipeline.device,
            dtype=dtype,
        )
        samples = pipeline.sample(
            samples,
            num_inference_steps=config.num_inference_steps,
            generator=torch.Generator(device).manual_seed(config.seed),
            verbose=config.verbose,
            use_neural_vocoder=config.use_neural_vocoder,
            num_griffin_lim_iters=config.num_griffin_lim_iters,
        )
        save_batch_samples(
            samples,
            offset=(batch_idx * config.batch_size),
            output_dir=config.output_dir,
            sample_rate=pipeline.transforms.sample_rate,
            audio_format=config.output_audio_format,
        )


def run_audio2audio(config: Audio2AudioConfig) -> None:
    # -------------------------------------------
    # Setup
    # -------------------------------------------
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)

    # -------------------------------------------
    # Load pipeline
    # -------------------------------------------
    pipeline = Pipeline.from_pretrained(config.ckpt_path).to(device).to(dtype)

    # -------------------------------------------
    # Prepare datamodule and dataloader
    # -------------------------------------------
    # Optionally compute the number of frames from duration
    num_frames = config.num_frames
    if config.duration is not None:
        num_frames = config.duration * pipeline.transforms.sample_rate
        num_frames = compute_divisible_length(
            num_frames,
            pipeline.transforms.hop_length,
            sum(pipeline.unet.has_resampling),
        )
    datamodule = AudioDataModule(
        config.data_dir,
        sample_rate=pipeline.transforms.sample_rate,
        num_frames=num_frames,
        load_random_slice=False,
        normalize_amplitude=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.train_dataloader()

    # -------------------------------------------
    # Run sampling
    # -------------------------------------------
    for batch_idx, batch in enumerate(
        tqdm(dataloader, desc="Audio2Audio", disable=(not config.verbose))
    ):
        samples = pipeline.sample(
            batch.to(device).to(dtype),
            num_inference_steps=config.num_inference_steps,
            strength=config.strength,
            generator=torch.Generator(device).manual_seed(config.seed),
            verbose=config.verbose,
            use_input_as_seed=True,
            use_neural_vocoder=config.use_neural_vocoder,
            num_griffin_lim_iters=config.num_griffin_lim_iters,
        )
        save_batch_samples(
            samples,
            offset=(batch_idx * config.batch_size),
            output_dir=config.output_dir,
            sample_rate=pipeline.transforms.sample_rate,
            audio_format=config.output_audio_format,
        )


def run_interpolation(config: InterpolationConfig) -> None:
    # -------------------------------------------
    # Setup
    # -------------------------------------------
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)

    # -------------------------------------------
    # Load pipeline
    # -------------------------------------------
    pipeline = Pipeline.from_pretrained(config.ckpt_path).to(device).to(dtype)

    # -------------------------------------------
    # Prepare datamodule and dataloader
    # -------------------------------------------
    # Optionally compute the number of frames from duration
    num_frames = config.num_frames
    if config.duration is not None:
        num_frames = config.duration * pipeline.transforms.sample_rate
        num_frames = compute_divisible_length(
            num_frames,
            pipeline.transforms.hop_length,
            sum(pipeline.unet.has_resampling),
        )
    first_datamodule = AudioDataModule(
        config.first_data_dir,
        sample_rate=pipeline.transforms.sample_rate,
        num_frames=num_frames,
        load_random_slice=False,
        normalize_amplitude=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )
    second_datamodule = AudioDataModule(
        config.second_data_dir,
        sample_rate=pipeline.transforms.sample_rate,
        num_frames=num_frames,
        load_random_slice=False,
        normalize_amplitude=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )
    first_datamodule.prepare_data()
    first_datamodule.setup()
    second_datamodule.prepare_data()
    second_datamodule.setup()

    first_dataloader = first_datamodule.train_dataloader()
    second_dataloader = second_datamodule.train_dataloader()

    assert len(first_dataloader) == len(
        second_dataloader
    ), "Samples should be equal in both directories"

    # -------------------------------------------
    # Run sampling
    # -------------------------------------------
    for batch_idx, (first_batch, second_batch) in enumerate(
        tqdm(
            zip(first_dataloader, second_dataloader),
            desc="Interpolation",
            disable=(not config.verbose),
        )
    ):
        samples = pipeline.interpolate(
            first_batch,
            second_batch,
            ratio=config.ratio,
            num_inference_steps=config.num_inference_steps,
            strength=config.strength,
            generator=torch.Generator(device).manual_seed(config.seed),
            verbose=config.verbose,
            use_neural_vocoder=config.use_neural_vocoder,
            num_griffin_lim_iters=config.num_griffin_lim_iters,
        )
        save_batch_samples(
            samples,
            offset=(batch_idx * config.batch_size),
            output_dir=config.output_dir,
            sample_rate=pipeline.transforms.sample_rate,
            audio_format=config.output_audio_format,
        )


def run_inpainting(config: InpaintingConfig) -> None:
    # -------------------------------------------
    # Setup
    # -------------------------------------------
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)

    # -------------------------------------------
    # Load pipeline
    # -------------------------------------------
    pipeline = Pipeline.from_pretrained(config.ckpt_path).to(device).to(dtype)

    # -------------------------------------------
    # Prepare datamodule and dataloader
    # -------------------------------------------
    # Optionally compute the number of frames from duration
    num_frames = config.num_frames
    if config.duration is not None:
        num_frames = config.duration * pipeline.transforms.sample_rate
        num_frames = compute_divisible_length(
            num_frames,
            pipeline.transforms.hop_length,
            sum(pipeline.unet.has_resampling),
        )
    datamodule = AudioDataModule(
        config.data_dir,
        sample_rate=pipeline.transforms.sample_rate,
        num_frames=num_frames,
        load_random_slice=False,
        normalize_amplitude=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.train_dataloader()

    # -------------------------------------------
    # Run sampling
    # -------------------------------------------
    assert len(dataloader) == math.ceil(
        len(config.masks) / config.batch_size
    ), "Number of masks should match the samples"

    for batch_idx, batch in enumerate(
        tqdm(dataloader, desc="Inpainting", disable=(not config.verbose))
    ):
        batch = batch.to(device).to(dtype)
        masks = generate_batch_audio_mask(
            config.masks[batch_idx : batch_idx + len(batch)],
            batch,
            pipeline.transforms.sample_rate,
        )

        samples = pipeline.inpaint(
            batch,
            masks,
            num_inference_steps=config.num_inference_steps,
            eta=config.eta,
            jump_length=config.jump_length,
            jump_n_sample=config.jump_n_sample,
            generator=torch.Generator(device).manual_seed(config.seed),
            verbose=config.verbose,
            use_neural_vocoder=config.use_neural_vocoder,
            num_griffin_lim_iters=config.num_griffin_lim_iters,
        )
        save_batch_samples(
            samples,
            offset=(batch_idx * config.batch_size),
            output_dir=config.output_dir,
            sample_rate=pipeline.transforms.sample_rate,
            audio_format=config.output_audio_format,
        )


def run_outpainting(config: OutpaintingConfig) -> None:
    # -------------------------------------------
    # Setup
    # -------------------------------------------
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype)

    # -------------------------------------------
    # Load pipeline
    # -------------------------------------------
    pipeline = Pipeline.from_pretrained(config.ckpt_path).to(device).to(dtype)

    # -------------------------------------------
    # Prepare datamodule and dataloader
    # -------------------------------------------
    # Optionally compute the number of frames from duration
    num_frames = config.num_frames
    if config.duration is not None:
        num_frames = config.duration * pipeline.transforms.sample_rate
        num_frames = compute_divisible_length(
            num_frames,
            pipeline.transforms.hop_length,
            sum(pipeline.unet.has_resampling),
        )
    datamodule = AudioDataModule(
        config.data_dir,
        sample_rate=pipeline.transforms.sample_rate,
        num_frames=num_frames,
        load_random_slice=False,
        normalize_amplitude=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=False,
    )
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.train_dataloader()

    # -------------------------------------------
    # Run sampling
    # -------------------------------------------
    for batch_idx, batch in enumerate(
        tqdm(dataloader, desc="Outpainting", disable=(not config.verbose))
    ):
        samples = pipeline.outpaint(
            batch.to(device).to(dtype),
            num_spans=config.num_spans,
            num_inference_steps=config.num_inference_steps,
            eta=config.eta,
            jump_length=config.jump_length,
            jump_n_sample=config.jump_n_sample,
            generator=torch.Generator(device).manual_seed(config.seed),
            verbose=config.verbose,
            use_neural_vocoder=config.use_neural_vocoder,
            num_griffin_lim_iters=config.num_griffin_lim_iters,
        )
        save_batch_samples(
            samples,
            offset=(batch_idx * config.batch_size),
            output_dir=config.output_dir,
            sample_rate=pipeline.transforms.sample_rate,
            audio_format=config.output_audio_format,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "task",
        help="task to run",
        choices=[
            "sampling",
            "audio2audio",
            "interpolation",
            "inpainting",
            "outpainting",
        ],
        type=str,
    )
    parser.add_argument("config_path", help="path to config file")
    args = parser.parse_args()

    file_config = OmegaConf.load(args.config_path)
    if args.task.lower() == "sampling":
        sampling_config = OmegaConf.structured(SamplingConfig)
        sampling_config = OmegaConf.merge(sampling_config, file_config)
        run_sampling(sampling_config)

    elif args.task.lower() == "audio2audio":
        audio2audio_config = OmegaConf.structured(Audio2AudioConfig)
        audio2audio_config = OmegaConf.merge(audio2audio_config, file_config)
        run_audio2audio(audio2audio_config)

    elif args.task.lower() == "interpolation":
        interpolation_config = OmegaConf.structured(InterpolationConfig)
        interpolation_config = OmegaConf.merge(InterpolationConfig, file_config)
        run_interpolation(interpolation_config)

    elif args.task.lower() == "inpainting":
        inpainting_config = OmegaConf.structured(InpaintingConfig)
        inpainting_config = OmegaConf.merge(InpaintingConfig, file_config)
        run_inpainting(inpainting_config)

    elif args.task.lower() == "outpainting":
        outpainting_config = OmegaConf.structured(OutpaintingConfig)
        outpainting_config = OmegaConf.merge(OutpaintingConfig, file_config)
        run_outpainting(outpainting_config)
