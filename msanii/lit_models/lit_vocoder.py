from typing import List, Tuple, Union

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, optim

from ..losses import SpectrogramLoss
from ..models import Vocoder
from ..transforms import Transforms
from .utils import log_samples


class LitVocoder(LightningModule):
    def __init__(
        self,
        transforms: Transforms,
        vocoder: Vocoder,
        sample_rate: int = 44_100,
        transforms_decay: float = 0.999,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.5, 0.999),
        lr_scheduler_start_factor: float = 1 / 3,
        lr_scheduler_iters: int = 500,
        sample_every_n_epochs: int = 10,
        num_samples: int = 4,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["transforms", "vocoder"])

        self.transforms = transforms
        self.vocoder = vocoder
        self.sample_rate = sample_rate
        self.transforms_decay = transforms_decay
        self.lr = lr
        self.betas = betas
        self.lr_scheduler_start_factor = lr_scheduler_start_factor
        self.lr_scheduler_iters = lr_scheduler_iters
        self.sample_every_n_epochs = sample_every_n_epochs
        self.num_samples = num_samples

        self.loss_fn = SpectrogramLoss()

    def training_step(
        self, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> Tensor:
        # Drop labels if present
        if isinstance(batch, list):
            batch = batch[0]

        # Reconstruct magnitude stft spectrogram from mel spectrograms
        mel_spectrograms = self.transforms(batch)
        pred_mag_spectrograms = self.vocoder(mel_spectrograms)

        # Compute & log spectral losses
        target_mag_spectrograms = self.transforms.spectrogram(batch)
        loss, loss_dict = self.loss_fn(pred_mag_spectrograms, target_mag_spectrograms)
        self.log_dict({"total_loss": loss})
        self.log_dict(loss_dict)

        # Update & log transforms parameters
        self.transforms.step()
        self.log_dict(self.transforms.params_dict)

        # Sample & log waveforms, spectrograms, distributions & audio samples
        if (self.current_epoch % self.sample_every_n_epochs == 0) and batch_idx == 0:
            self.__sample_and_log_samples(
                batch, mel_spectrograms, pred_mag_spectrograms
            )

        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.vocoder.parameters(), lr=self.lr, betas=self.betas)
        sched = optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self.lr_scheduler_start_factor,
            total_iters=self.lr_scheduler_iters,
        )
        sched = {"scheduler": sched, "interval": "step"}

        return [opt], [sched]

    @torch.no_grad()
    def __sample_and_log_samples(
        self, waveforms: Tensor, mel_spectrograms: Tensor, pred_mag_spectrograms: Tensor
    ) -> None:
        # Ensure the number of samples does not exceed the batch size
        num_samples = min(self.num_samples, waveforms.shape[0])

        # Ground truth samples
        log_samples(
            self.logger,
            waveforms[:num_samples],
            mel_spectrograms[:num_samples],
            self.sample_rate,
            "ground_truth",
        )

        # Neural vocoder samples
        pred_batch = self.transforms.griffin_lim(
            pred_mag_spectrograms[:num_samples], length=waveforms.shape[-1]
        )
        pred_mel_spectrograms = self.transforms(pred_batch)
        log_samples(
            self.logger,
            pred_batch,
            pred_mel_spectrograms,
            self.sample_rate,
            "neural_vocoder",
        )

        # Direct reconstruction samples
        pred_batch = self.transforms.inverse_transform(
            mel_spectrograms[:num_samples], length=waveforms.shape[-1]
        )
        pred_mel_spectrograms = self.transforms(pred_batch)
        log_samples(
            self.logger,
            pred_batch,
            pred_mel_spectrograms,
            self.sample_rate,
            "direct_reconstruction",
        )
