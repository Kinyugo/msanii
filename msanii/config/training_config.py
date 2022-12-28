from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import MISSING


@dataclass
class AudioDataModuleConfig:
    _target_: str = "msanii.data.AudioDataModule"
    data_dir: str = MISSING
    sample_rate: int = 44_100
    num_frames: Optional[int] = None
    load_random_slice: bool = False
    normalize_amplitude: bool = True
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = False


@dataclass
class TransformsConfig:
    _target_: str = "msanii.transforms.Transforms"
    sample_rate: int = 44_100
    n_fft: int = 2048
    win_length: Optional[int] = None
    hop_length: Optional[int] = None
    n_mels: int = 128
    feature_range: Tuple[float, float] = (-1.0, 1.0)
    momentum: float = 1e-3
    eps: float = 1e-5
    clip: bool = True
    num_griffin_lim_iters: int = 50
    griffin_lim_momentum: float = 0.99


@dataclass
class VocoderConfig:
    _target_: str = "msanii.models.Vocoder"
    n_fft: int = 2048
    n_mels: int = 128
    d_model: int = 256
    d_hidden_factor: int = 4


@dataclass
class UNetConfig:
    _target_: str = "msanii.models.UNet"
    d_freq: int = 128
    d_base: int = 256
    d_hidden_factor: int = 4
    d_multipliers: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1])
    d_timestep: int = 128
    dilations: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1])
    n_heads: int = 8
    has_attention: List[bool] = field(
        default_factory=lambda: [False, False, False, False, False, True, True]
    )
    has_resampling: List[bool] = field(
        default_factory=lambda: [True, True, True, True, True, True, False]
    )
    n_block_layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2, 2, 2, 2])


@dataclass
class SchedulerConfig:
    _target_: str = "diffusers.DDIMScheduler"
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    _kwargs_: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class LitVocoderConfig:
    _target_: str = "msanii.lit_models.LitVocoder"
    sample_rate: int = 44_100
    transforms_decay: float = 0.999
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    lr_scheduler_start_factor: float = 1 / 3
    lr_scheduler_iters: int = 500
    sample_every_n_epochs: int = 10
    num_samples: int = 4


@dataclass
class LitDiffusionConfig:
    _target_: str = "msanii.lit_models.LitDiffusion"
    sample_rate: int = 44_100
    transforms_decay: float = 0.999
    ema_decay: float = 0.995
    ema_start_step: int = 2000
    ema_update_every: int = 10
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    lr_scheduler_start_factor: float = 1 / 3
    lr_scheduler_iters: int = 500
    sample_every_n_epochs: int = 10
    num_samples: int = 4
    num_inference_steps: int = 20


@dataclass
class WandbLoggerConfig:
    _target_: str = "lightning.pytorch.loggers.WandbLogger"
    save_dir: str = "logs"
    project: str = "msanii"
    name: Optional[str] = None
    job_type: Optional[str] = "train"
    log_model: Union[str, bool] = True
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    save_code: Optional[bool] = True
    offline: bool = False
    _kwargs_: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class ModelCheckpointConfig:
    _target_: str = "lightning.pytorch.callbacks.ModelCheckpoint"
    dirpath: Optional[str] = None
    save_last: Optional[bool] = True
    verbose: bool = False
    mode: str = "min"
    _kwargs_: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class TrainerConfig:
    _target_: str = "lightning.Trainer"
    accelerator: Optional[str] = "auto"
    accumulate_grad_batches: int = 1
    devices: Optional[Union[int, str]] = None
    default_root_dir: Optional[str] = None
    detect_anomaly: bool = False
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    limit_train_batches: Optional[Union[int, float]] = 1.0
    log_every_n_steps: int = 10
    precision: Union[int, str] = 32
    max_epochs: Optional[int] = 6
    max_steps: int = -1
    weights_save_path: Optional[str] = None
    fast_dev_run: bool = False
    _kwargs_: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class VocoderTrainingConfig:
    datamodule: AudioDataModuleConfig = field(default_factory=AudioDataModuleConfig)
    transforms: TransformsConfig = field(default_factory=TransformsConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    lit_vocoder: LitVocoderConfig = field(default_factory=LitVocoderConfig)
    wandb_logger: WandbLoggerConfig = field(default_factory=WandbLoggerConfig)
    model_checkpoint: ModelCheckpointConfig = field(
        default_factory=ModelCheckpointConfig
    )
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    skip_training: bool = False
    resume_ckpt_path: Optional[str] = None


@dataclass
class DiffusionTrainingConfig:
    datamodule: AudioDataModuleConfig = field(default_factory=AudioDataModuleConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    lit_diffusion: LitDiffusionConfig = field(default_factory=LitDiffusionConfig)
    wandb_logger: WandbLoggerConfig = field(default_factory=WandbLoggerConfig)
    model_checkpoint: ModelCheckpointConfig = field(
        default_factory=ModelCheckpointConfig
    )
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    skip_training: bool = False
    resume_ckpt_path: Optional[str] = None


@dataclass
class TrainingConfig:
    vocoder: VocoderTrainingConfig = field(default_factory=VocoderTrainingConfig)
    diffusion: DiffusionTrainingConfig = field(default_factory=DiffusionTrainingConfig)

    seed: int = 0
    pipeline_wandb_name: str = "msanii_pipeline"
    pipeline_ckpt_path: str = "checkpoints/msanii.pt"
