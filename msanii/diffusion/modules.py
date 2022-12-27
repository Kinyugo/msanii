from typing import Any, Optional, Union

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, RePaintScheduler
from torch import Tensor, nn
from tqdm.autonotebook import tqdm

from .utils import noise_like, sequential_mask


class DiffusionModule(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler],
    ) -> None:
        super().__init__()

        self.eps_model = eps_model
        self.scheduler = scheduler

        self.scheduler.set_timesteps(self.scheduler.num_train_timesteps)

    @property
    def device(self) -> torch.device:
        next(self.eps_model.parameters()).device


class Sampler(DiffusionModule):
    def forward(
        self,
        x: Tensor,
        num_inference_steps: Optional[int] = None,
        timestep: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        **kwargs: Any
    ) -> Tensor:
        # Start sampling from the last diffusion timestep
        if timestep is None:
            timestep = self.scheduler.num_train_timesteps - 1
        # Iterate over all timesteps from the starting timestep
        if num_inference_steps is None:
            num_inference_steps = timestep + 1
        # Make sure we don't sample out of range values
        num_inference_steps = min(num_inference_steps, timestep + 1)

        # Adjust timesteps to match the number of sampling steps and starting timestep
        step = (timestep + 1) // num_inference_steps
        timesteps = torch.arange(
            start=0, end=timestep + 1, step=step, device=self.device
        ).flipud()

        for t in tqdm(timesteps, disable=(not verbose)):
            # Prepare a batch of timesteps
            batch_t = torch.full(
                (x.shape[0],), timestep, dtype=torch.long, device=x.device
            )
            # Get model estimate
            eps = self.eps_model(x, batch_t)
            # Compute the denoised sample
            x = self.scheduler.step(
                eps, t, x, generator=generator, **kwargs
            ).prev_sample

        return x


class Interpolater(DiffusionModule):
    def __init__(
        self,
        eps_model: nn.Module,
        scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler],
    ) -> None:
        super().__init__(eps_model, scheduler)

        self.sampler = Sampler(self.eps_model, self.scheduler)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        ratio: float = 0.5,
        num_inference_steps: Optional[int] = None,
        timestep: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        **kwargs: Any
    ) -> Tensor:
        # Start interpolation from complete noise
        if timestep is None:
            timestep = self.scheduler.num_train_timesteps - 1

        # Prepare a batch of timesteps
        timesteps = torch.full(
            (x1.shape[0],), timestep, dtype=torch.long, device=x1.device
        )

        # Add noise up to the starting timestep
        x1_noisy = self.scheduler.add_noise(x1, noise_like(x1, generator), timesteps)
        x2_noisy = self.scheduler.add_noise(x2, noise_like(x2, generator), timesteps)

        # Interpolate between the two samples in latent/noisy space
        x = ratio * x1_noisy + (1 - ratio) * x2_noisy
        x = self.sampler(x, num_inference_steps, timestep, generator, verbose, **kwargs)

        return x


class Inpainter(DiffusionModule):
    def __init__(
        self,
        eps_model: nn.Module,
        scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler],
    ) -> None:
        super().__init__(eps_model, scheduler)

        self.repaint_scheduler = RePaintScheduler.from_config(self.scheduler.config)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        num_inference_steps: Optional[int] = None,
        eta: float = 0.0,
        jump_length=10,
        jump_n_sample=10,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        **kwargs: Any
    ) -> Tensor:
        # Iterate over all timesteps
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_train_timesteps

        # Adjust scheduler inference steps
        self.repaint_scheduler.set_timesteps(
            num_inference_steps, jump_length, jump_n_sample, self.device
        )
        self.repaint_scheduler.eta = eta

        # Start inpainting from complete noise
        x_inpainted = noise_like(x)

        t_last = self.repaint_scheduler.timesteps[0] + 1
        for t in tqdm(self.repaint_scheduler.timesteps, disable=(not verbose)):
            if t < t_last:
                # Prepare a batch of timesteps
                batch_t = torch.full(
                    (x.shape[0],), t, dtype=torch.long, device=x.device
                )
                # Get model estimate
                eps = self.eps_model(x, batch_t)
                # Compute the denoised sample
                x_inpainted = self.repaint_scheduler.step(
                    eps, t, x_inpainted, x, mask, generator, **kwargs
                ).prev_sample
            else:
                x_inpainted = self.repaint_scheduler.undo_step(
                    x_inpainted, t_last, generator
                )
            t_last = t

        return x_inpainted


class Outpainter(DiffusionModule):
    def __init__(
        self,
        eps_model: nn.Module,
        scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler],
    ) -> None:
        super().__init__(eps_model, scheduler)

        self.inpainter = Inpainter(eps_model, scheduler)

    def forward(
        self,
        x: Tensor,
        num_spans: int = 1,
        num_inference_steps: Optional[int] = None,
        eta: float = 0.0,
        jump_length=10,
        jump_n_sample=10,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
        **kwargs: Any
    ) -> Tensor:
        half_length = x.shape[-1] // 2

        spans = list(x.chunk(chunks=2, dim=-1))
        # Inpaint second half from first half
        inpaint = torch.zeros_like(x)
        inpaint[..., :half_length] = x[..., half_length:]
        inpaint_mask = sequential_mask(like=x, start=half_length).to(x.dtype)

        for _ in range(num_spans):
            # Inpaint second half
            span = self.inpainter(
                inpaint,
                inpaint_mask,
                num_inference_steps,
                eta,
                jump_length,
                jump_n_sample,
                generator,
                verbose,
                **kwargs
            )
            # Replace first half with generated second half
            second_half = span[..., half_length:]
            inpaint[..., :half_length] = second_half
            # Save generated span
            spans.append(second_half)

        return torch.cat(spans, dim=-1)
