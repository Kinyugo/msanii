import wandb
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn

from ..utils import plot_distribution, plot_spectrogram, plot_waveform


def log_waveform(
    logger: WandbLogger, waveform: Tensor, sample_rate: int, id: str, caption: str = ""
) -> None:
    logger.log(
        {
            f"{id}_{idx}": plot_waveform(waveform[idx], sample_rate, caption)
            for idx in waveform.shape[0]
        }
    )


def log_spectrogram(
    logger: WandbLogger, spectrogram: Tensor, id: str, caption: str = ""
) -> None:
    logger.log(
        {
            f"{id}_{idx}": plot_spectrogram(spectrogram[idx], caption)
            for idx in spectrogram.shape[0]
        }
    )


def log_distribution(
    logger: WandbLogger, x: Tensor, id: str, caption: str = ""
) -> None:
    logger.log(
        {f"{id}_{idx}": plot_distribution(x[idx], caption) for idx in x.shape[0]}
    )


def log_audio(
    logger: WandbLogger, audio: Tensor, sample_rate: int, id: str, caption: str = ""
) -> None:
    audio = rearrange(audio, "b c t -> b t c").detach().cpu().numpy()
    logger.log(
        {
            f"{id}_{idx}": wandb.Audio(audio[idx], sample_rate, caption)
            for idx in audio.shape[0]
        }
    )


def log_samples(
    logger: WandbLogger,
    waveform: Tensor,
    spectrogram: Tensor,
    sample_rate: int,
    tag: str,
) -> None:
    log_spectrogram(logger, spectrogram, f"spectrogram/{tag}", tag)
    log_distribution(logger, spectrogram, f"distribution/{tag}", tag)
    log_waveform(logger, waveform, sample_rate, f"waveform/{tag}", tag)
    log_audio(logger, waveform, sample_rate, f"audio/{tag}", tag)


def update_ema_model(src_model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    for ema_param, src_param in zip(ema_model.parameters(), src_model.parameters()):
        if ema_param.data is None:
            ema_param.data = src_param.data
        else:
            ema_param.data = decay * ema_param.data + (1 - decay) * src_param.data
