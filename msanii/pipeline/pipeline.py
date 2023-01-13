from typing import Any, Callable, Optional, Union

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from einops import repeat
from torch import Tensor, nn
from torch.nn import functional as F
from typing_extensions import Self

from ..config import from_config
from ..diffusion import Inpainter, Interpolater, Outpainter, Sampler
from ..diffusion.utils import noise_like
from ..models import UNet, Vocoder
from ..transforms import Transforms


class Pipeline(nn.Module):
    def __init__(
        self,
        transforms: Transforms,
        vocoder: Vocoder,
        unet: UNet,
        scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler],
    ) -> None:
        super().__init__()

        self.transforms = transforms
        self.vocoder = vocoder
        self.unet = unet
        self.scheduler = scheduler

    @property
    def dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.unet.parameters()).device

    @torch.no_grad()
    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        x: Tensor,
        num_inference_steps: Optional[int] = None,
        strength: float = 1.0,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        use_input_as_seed: bool = False,
        use_neural_vocoder: bool = True,
        num_griffin_lim_iters: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        # Initialize the sampler
        sampler = Sampler(self.unet, self.scheduler).to(self.device)

        num_frames = x.shape[-1]

        # Start from an initial sample
        if use_input_as_seed:
            # Add noise to the initial sample up to the last timestep
            timesteps, _ = sampler.compute_timesteps(
                num_inference_steps, strength, x.device
            )

            # Convert waveform to mel and add noise to the starting timestep
            batch_t = torch.full(
                (x.shape[0],), timesteps[0], dtype=torch.long, device=self.device
            )
            x = self.transforms(x)
            x = sampler.scheduler.add_noise(x, noise_like(x, generator), batch_t)

        # Start from random noise
        else:
            x = noise_like(self.transforms(x), generator)

        # Denoise the samples
        x = sampler(x, num_inference_steps, strength, generator, verbose, **kwargs)

        return self.__vocode(x, use_neural_vocoder, num_frames, num_griffin_lim_iters)

    @torch.no_grad()
    def interpolate(
        self,
        x1: Tensor,
        x2: Tensor,
        ratio: float = 0.5,
        num_inference_steps: Optional[int] = None,
        strength: float = 1.0,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        use_neural_vocoder: bool = True,
        num_griffin_lim_iters: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        # Initialize the interpolater
        interpolater = Interpolater(self.unet, self.scheduler).to(self.device)

        # Transform inputs into mel spectrograms
        num_frames = x1.shape[-1]
        x1 = self.transforms(x1)
        x2 = self.transforms(x2)

        # Interpolate in the melspace
        x = interpolater(
            x1, x2, ratio, num_inference_steps, strength, generator, verbose, **kwargs
        )

        return self.__vocode(x, use_neural_vocoder, num_frames, num_griffin_lim_iters)

    @torch.no_grad()
    def inpaint(
        self,
        x: Tensor,
        mask: Tensor,
        num_inference_steps: Optional[int] = None,
        eta: float = 0.0,
        jump_length: int = 10,
        jump_n_sample: int = 10,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        use_neural_vocoder: bool = True,
        num_griffin_lim_iters: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        # Initialize the inpainter
        inpainter = Inpainter(self.unet, self.scheduler).to(self.device)

        # Transform input into mel spectrogram
        num_frames = x.shape[-1]
        x = self.transforms(x)

        # Rescale the mask to match shape of the spectrogram
        mask = F.interpolate(mask, size=x.shape[-1])
        mask = repeat(mask, "b c t -> b c f t", f=x.shape[-2])

        x = inpainter(
            x,
            mask,
            num_inference_steps,
            eta,
            jump_length,
            jump_n_sample,
            generator,
            verbose,
            **kwargs,
        )

        return self.__vocode(x, use_neural_vocoder, num_frames, num_griffin_lim_iters)

    @torch.no_grad()
    def outpaint(
        self,
        x: Tensor,
        num_spans: int = 2,
        num_inference_steps: Optional[int] = None,
        eta: float = 0.0,
        jump_length=10,
        jump_n_sample=10,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        use_neural_vocoder: bool = True,
        num_griffin_lim_iters: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        # Initialize the inpainter
        outpainter = Outpainter(self.unet, self.scheduler).to(self.device)

        # Transform input into mel spectrogram
        num_frames = x.shape[-1]
        x = self.transforms(x)

        x = outpainter(
            x,
            num_spans,
            num_inference_steps,
            eta,
            jump_length,
            jump_n_sample,
            generator,
            verbose,
            **kwargs,
        )

        # Compute the new number of frames
        num_frames = num_frames + ((num_frames // 2) * num_spans)
        x = self.__vocode(x, use_neural_vocoder, None, num_griffin_lim_iters)

        return x[..., :num_frames]

    @torch.no_grad()
    def save_pretrained(self, ckpt_path: str) -> None:
        checkpoint = {
            "transforms_config": self.transforms.config,
            "vocoder_config": self.vocoder.config,
            "unet_config": self.unet.config,
            "scheduler_config": self.scheduler.config,
            "transforms_state_dict": self.transforms.state_dict(),
            "vocoder_state_dict": self.vocoder.state_dict(),
            "unet_state_dict": self.unet.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str,
        scheduler_class: Union[
            DDIMScheduler, DPMSolverMultistepScheduler
        ] = DDIMScheduler,
        device: Optional[torch.device] = None,
    ) -> Self:
        checkpoint = torch.load(ckpt_path)

        transforms = Pipeline._load_from_checkpoint(
            checkpoint, "transforms", Transforms
        )
        vocoder = Pipeline._load_from_checkpoint(checkpoint, "vocoder", Vocoder)
        unet = Pipeline._load_from_checkpoint(checkpoint, "unet", UNet)
        scheduler = from_config(checkpoint["scheduler_config"], scheduler_class)

        return cls(transforms, vocoder, unet, scheduler).to(device)

    def __vocode(
        self,
        x: Tensor,
        use_neural_vocoder: bool = True,
        num_frames: Optional[int] = None,
        num_griffin_lim_iters: Optional[int] = None,
    ) -> Tensor:
        if use_neural_vocoder:
            return self.transforms.griffin_lim(
                self.vocoder(x), length=num_frames, num_iters=num_griffin_lim_iters
            )

        return self.transforms(
            x,
            inverse=True,
            length=num_frames,
            num_griffin_lim_iters=num_griffin_lim_iters,
        )

    @staticmethod
    def _load_from_checkpoint(checkpoint, prefix: str, target: Callable) -> Any:
        target_instance = from_config(checkpoint[f"{prefix}_config"], target)
        target_instance.load_state_dict(checkpoint[f"{prefix}_state_dict"])

        return target_instance


if __name__ == "__main__":
    pipeline = Pipeline(Transforms(), UNet(), Vocoder(), DDIMScheduler())
    x = torch.randn((1, 2, 523_264))
    x = pipeline.outpaint(x, num_spans=2, num_inference_steps=2, verbose=True)
    print("samples", x.shape)
