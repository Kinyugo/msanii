from argparse import ArgumentParser
from typing import Tuple, Union

import lightning as L
import matplotlib
from diffusers import DDIMPipeline, DPMSolverMultistepScheduler
from lightning.pytorch.callbacks import RichModelSummary, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from ..config import (
    DiffusionTrainingConfig,
    TrainingConfig,
    VocoderTrainingConfig,
    from_config,
)
from ..models import UNet, Vocoder
from ..pipeline import Pipeline
from ..transforms import Transforms
from ..utils import clone_model_parameters


def run_vocoder_training(config: VocoderTrainingConfig) -> Tuple[Transforms, Vocoder]:
    # -------------------------------------------
    # Data & Transforms
    # -------------------------------------------
    datamodule = from_config(config.datamodule)
    transforms = from_config(config.transforms)

    # -----------------------------------------
    # Model
    # ------------------------------------------
    vocoder = from_config(config.vocoder)

    # -----------------------------------------
    # Lit Model
    # ------------------------------------------
    lit_vocoder = from_config(
        config.lit_vocoder, transforms=transforms, vocoder=vocoder
    )

    # -----------------------------------------
    # Logger & Callbacks
    # ------------------------------------------
    wandb_logger = from_config(config.wandb_logger)
    model_checkpoint = from_config(config.model_checkpoint)
    callbacks = [model_checkpoint, RichModelSummary(), TQDMProgressBar()]

    # -----------------------------------------
    # Trainer
    # ------------------------------------------
    trainer = from_config(config.trainer, logger=wandb_logger, callbacks=callbacks)

    # -----------------------------------------
    # Run Training
    # ------------------------------------------
    # Save config to wandb
    wandb_logger.experiment.config.update(dict(config))

    # Optionally run training
    if not config.skip_training:
        trainer.fit(
            lit_vocoder, datamodule=datamodule, ckpt_path=config.resume_ckpt_path
        )

    # Terminate wandb run
    wandb_logger.experiment.finish()

    return lit_vocoder.transforms, lit_vocoder.vocoder


def run_diffusion_training(
    config: DiffusionTrainingConfig, transforms: Transforms, vocoder: Vocoder
) -> Tuple[
    Transforms, Vocoder, Union[DDIMPipeline, DPMSolverMultistepScheduler], WandbLogger
]:
    # -------------------------------------------
    # Data
    # -------------------------------------------
    datamodule = from_config(config.datamodule)

    # -----------------------------------------
    # Models
    # ------------------------------------------
    unet = from_config(config.unet)
    ema_unet = from_config(config.unet)
    ema_unet = clone_model_parameters(unet, ema_unet)

    # -----------------------------------------
    # Scheduler
    # ------------------------------------------
    scheduler = from_config(config.scheduler)

    # -----------------------------------------
    # Lit Model
    # ------------------------------------------
    lit_diffusion = from_config(
        config.lit_diffusion,
        transforms=transforms,
        vocoder=vocoder,
        unet=unet,
        ema_unet=ema_unet,
        scheduler=scheduler,
    )

    # -----------------------------------------
    # Logger & Callbacks
    # ------------------------------------------
    wandb_logger = from_config(config.wandb_logger, reinit=True)
    model_checkpoint = from_config(config.model_checkpoint)
    callbacks = [model_checkpoint, RichModelSummary(), TQDMProgressBar()]

    # -----------------------------------------
    # Trainer
    # ------------------------------------------
    trainer = from_config(config.trainer, logger=wandb_logger, callbacks=callbacks)

    # -----------------------------------------
    # Run Training
    # ------------------------------------------
    # Save config to wandb
    wandb_logger.experiment.config.update(dict(config))

    # Optionally run training
    if not config.skip_training:
        trainer.fit(
            lit_diffusion, datamodule=datamodule, ckpt_path=config.resume_ckpt_path
        )

    return (
        lit_diffusion.transforms,
        lit_diffusion.ema_unet,
        lit_diffusion.scheduler,
        wandb_logger,
    )


def run_training(config: TrainingConfig) -> None:
    # -------------------------------------------
    # Reproducibility
    # -------------------------------------------
    L.seed_everything(config.seed)

    # -------------------------------------------
    # Configure Matplotlib
    # -------------------------------------------
    # Prevents pixelated fonts on figures
    matplotlib.use("webagg")
    matplotlib.style.use(["seaborn", "fast"])

    # -------------------------------------------
    # Train Vocoder
    # -------------------------------------------
    transforms, vocoder = run_vocoder_training(config.vocoder)

    # -------------------------------------------
    # Train Diffusion
    # -------------------------------------------
    transforms, unet, scheduler, wandb_logger = run_diffusion_training(
        config.diffusion, transforms, vocoder
    )

    # -------------------------------------------
    # Save Pipeline Checkpoint
    # -------------------------------------------
    pipeline = Pipeline(transforms, vocoder, unet, scheduler)
    pipeline.save_pretrained(config.pipeline_ckpt_path)

    # -------------------------------------------
    # Log checkpoint to Wandb
    # -------------------------------------------
    artifact = wandb_logger.experiment.Artifact(
        config.pipeline_wandb_name, type="model"
    )
    artifact.add_file(config.pipeline_ckpt_path)
    wandb_logger.experiment.log_artifact(artifact)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_path", help="path to config file", type=str)
    args = parser.parse_args()

    default_training_config = OmegaConf.structured(TrainingConfig)
    file_training_config = OmegaConf.load(args.config_path)
    training_config = OmegaConf.merge(default_training_config, file_training_config)

    run_training(training_config)
