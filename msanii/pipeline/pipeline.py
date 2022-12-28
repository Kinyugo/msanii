from typing import Any, Dict, Optional, Union

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from einops import repeat
from torch import Tensor, nn
from torch.nn import functional as F
from typing_extensions import Self

from ..config import TrainingConfig, from_config
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
    def device(self) -> torch.device:
        next(self.unet.parameters()).device

    @torch.no_grad()
    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        x: Tensor,
        num_inference_steps: Optional[int] = None,
        timestep: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        use_input_as_seed: bool = False,
        use_neural_vocoder: bool = True,
        **kwargs: Any,
    ) -> Tensor:
        # Initialize the sampler
        sampler = Sampler(self.unet, self.scheduler).to(self.device)

        num_frames = x.shape[-1]

        # Start from an initial sample
        if use_input_as_seed:
            # Add noise to the initial sample up to the last timestep
            if timestep is None:
                timestep = sampler.scheduler.num_train_timesteps - 1

            # Convert waveform to mel and add noise to the starting timestep
            batch_t = torch.full(
                (x.shape[0],), timestep, dtype=torch.long, device=self.device
            )
            x = sampler.scheduler.add_noise(
                self.transforms(x), noise_like(x, generator), batch_t
            )

        # Start from random noise
        else:
            x = torch.randn_like(self.transforms(x))

        # Denoise the samples
        x = sampler(x, num_inference_steps, timestep, generator, verbose, **kwargs)

        return self.__vocode(x, use_neural_vocoder, num_frames)

    @torch.no_grad()
    def interpolate(
        self,
        x1: Tensor,
        x2: Tensor,
        ratio: float = 0.5,
        num_inference_steps: Optional[int] = None,
        timestep: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        use_neural_vocoder: bool = True,
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
            x1, x2, ratio, num_inference_steps, timestep, generator, verbose, **kwargs
        )

        return self.__vocode(x, use_neural_vocoder, num_frames)

    @torch.no_grad()
    def inpaint(
        self,
        x: Tensor,
        mask: Tensor,
        num_inference_steps: Optional[int] = None,
        eta: float = 0.0,
        jump_length=10,
        jump_n_sample=10,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        use_neural_vocoder: bool = True,
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

        return self.__vocode(x, use_neural_vocoder, num_frames)

    @torch.no_grad()
    def outpaint(
        self,
        x: Tensor,
        num_spans: int = 1,
        num_inference_steps: Optional[int] = None,
        eta: float = 0.0,
        jump_length=10,
        jump_n_sample=10,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        use_neural_vocoder: bool = True,
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
        x = self.__vocode(x, use_neural_vocoder, None)

        return x[..., :num_frames]

    @torch.no_grad()
    def save_pretrained(self, ckpt_path: str, config: TrainingConfig) -> None:
        checkpoint = {
            "config": config,
            "transforms": self.transforms.state_dict(),
            "vocoder": self.vocoder.state_dict(),
            "unet": self.unet.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_path: str,
        device: Optional[torch.device] = None,
    ) -> Self:
        checkpoint = torch.load(ckpt_path)

        # Initialize models
        config = checkpoint["config"]
        transforms = from_config(config.vocoder.transforms, Transforms)
        vocoder = from_config(config.vocoder.vocoder, Vocoder)
        unet = from_config(config.diffusion.unet, UNet)
        scheduler = from_config(config.diffusion.scheduler)

        # Load model state
        transforms.load_state_dict(checkpoint["transforms"])
        vocoder.load_state_dict(checkpoint["vocoder"])
        unet.load_state_dict(checkpoint["unet"])

        return cls(transforms, vocoder, unet, scheduler).to(device)

    def __vocode(
        self,
        x: Tensor,
        use_neural_vocoder: bool = True,
        num_frames: Optional[int] = None,
    ) -> Tensor:
        if use_neural_vocoder:
            return self.transforms.griffin_lim(self.vocoder(x), length=num_frames)
        return self.transforms(x, inverse=True, length=num_frames)


if __name__ == "__main__":
    pipeline = Pipeline(Transforms(), UNet(), Vocoder(), DDIMScheduler())
    x = torch.randn((1, 2, 523_264))
    x = pipeline.outpaint(x, num_spans=2, num_inference_steps=2, verbose=True)
    print("samples", x.shape)
