from typing import List, Tuple, Union

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from lightning.pytorch import LightningModule
from torch import Tensor, nn, optim

from ..diffusion import Sampler
from ..models import UNet, Vocoder
from ..transforms import Transforms
from ..utils import freeze_model
from .utils import log_samples, update_ema_model


class LitDiffusion(LightningModule):
    def __init__(
        self,
        transforms: Transforms,
        vocoder: Vocoder,
        unet: UNet,
        ema_unet: UNet,
        scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler],
        sample_rate: int = 44_100,
        transforms_decay: float = 0.999,
        ema_decay: float = 0.995,
        ema_start_step: int = 2000,
        ema_update_every: int = 10,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.5, 0.999),
        lr_scheduler_start_factor: float = 1 / 3,
        lr_scheduler_iters: int = 500,
        sample_every_n_epochs: int = 10,
        num_samples: int = 4,
        num_inference_steps: int = 20,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            ignore=["transforms", "vocoder", "unet", "ema_unet", "scheduler"]
        )

        self.transforms = transforms
        self.vocoder = freeze_model(vocoder)
        self.unet = unet
        self.ema_unet = ema_unet
        self.scheduler = scheduler
        self.sample_rate = sample_rate
        self.transforms_decay = transforms_decay
        self.ema_decay = ema_decay
        self.ema_start_step = ema_start_step
        self.ema_update_every = ema_update_every
        self.lr = lr
        self.betas = betas
        self.lr_scheduler_start_factor = lr_scheduler_start_factor
        self.lr_scheduler_iters = lr_scheduler_iters
        self.sample_every_n_epochs = sample_every_n_epochs
        self.num_samples = num_samples
        self.num_inference_steps = num_inference_steps

        self.loss_fn = nn.L1Loss()
        self.sampler = Sampler(self.ema_unet, self.scheduler)

    def training_step(
        self, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> Tensor:
        # Drop labels if present
        if isinstance(batch, list):
            batch = batch[0]

        # Transform batch to mel spectrograms & add noise
        mel_spectrograms = self.transforms(batch)
        noise = torch.randn_like(mel_spectrograms)
        timesteps = torch.randint(
            self.scheduler.num_train_timesteps,
            size=(batch.shape[0],),
            dtype=torch.long,
            device=self.device,
        )
        noisy_mel_spectrograms = self.scheduler.add_noise(
            mel_spectrograms, noise, timesteps
        )
        # Predict added noise
        pred_noise = self.unet(noisy_mel_spectrograms, timesteps)

        # Compute & log reconstruction loss
        loss = self.loss_fn(pred_noise, noise)
        self.log_dict({"total_loss": loss})

        # Update & log transforms parameters
        self.transforms.step()
        self.log_dict(self.transforms.params_dict)

        # Update moving average model parameters
        if self.global_step % self.ema_update_every == 0:
            self.__update_ema_model()

        # Sample & log waveforms, spectrograms, distributions & audio samples
        if (self.current_epoch % self.sample_every_n_epochs == 0) and batch_idx == 0:
            self.__sample_and_log_samples(batch, mel_spectrograms)

        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.unet.parameters(), lr=self.lr, betas=self.betas)
        sched = optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self.lr_scheduler_start_factor,
            total_iters=self.lr_scheduler_iters,
        )
        sched = {"scheduler": sched, "interval": "step"}

        return [opt], [sched]

    @torch.no_grad()
    def __update_ema_model(self) -> None:
        # Copy source model parameters
        if self.global_step < self.ema_start_step:
            self.ema_unet.load_state_dict(self.unet.state_dict())

        # Update ema model parameters with the moving average
        else:
            update_ema_model(self.unet, self.ema_unet, self.ema_decay)

    @torch.no_grad()
    def __sample_and_log_samples(
        self, waveforms: Tensor, mel_spectrograms: Tensor
    ) -> None:
        # Ensure the number of samples does not exceed the batch size
        num_samples = min(self.num_samples, waveforms.shape[0])

        # Generate samples
        noise = torch.randn_like(mel_spectrograms)[:num_samples]
        sample_mel_spectrograms = self.sampler(
            noise, num_inference_steps=self.num_inference_steps, verbose=True
        )

        # Ground truth samples
        log_samples(
            self.logger,
            waveforms[:num_samples],
            mel_spectrograms[:num_samples],
            self.sample_rate,
            "ground_truth",
        )

        # Neural vocoder samples
        sample_waveforms = self.transforms.griffin_lim(
            self.vocoder(sample_mel_spectrograms), length=waveforms.shape[-1]
        )
        log_samples(
            self.logger,
            sample_waveforms,
            sample_mel_spectrograms,
            self.sample_rate,
            "neural_vocoder",
        )

        # Direct reconstruction samples
        sample_waveforms = self.transforms.inverse_transform(
            sample_mel_spectrograms, length=waveforms.shape[-1]
        )
        log_samples(
            self.logger,
            sample_waveforms,
            sample_mel_spectrograms,
            self.sample_rate,
            "direct_reconstruction",
        )
