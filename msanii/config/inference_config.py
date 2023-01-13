from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING


@dataclass
class SharedConfig:
    ckpt_path: str = MISSING
    output_dir: str = MISSING

    batch_size: int = 4
    num_frames: int = 8_387_584
    output_audio_format: str = "wav"

    seed: int = 0
    device: str = "cpu"
    dtype: str = "float"

    num_inference_steps: Optional[int] = None
    verbose: bool = False
    use_neural_vocoder: bool = True
    num_griffin_lim_iters: Optional[int] = None

    _kwargs_: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class SamplingConfig(SharedConfig):
    channels: int = 2
    num_samples: int = 16


@dataclass
class Audio2AudioConfig(SharedConfig):
    data_dir: str = MISSING
    num_workers: int = 4
    pin_memory: bool = False

    strength: float = 1.0


@dataclass
class InterpolationConfig(SharedConfig):
    first_data_dir: str = MISSING
    second_data_dir: str = MISSING
    num_workers: int = 4
    pin_memory: bool = False

    ratio: float = 0.5
    strength: float = 1.0


@dataclass
class InpaintingConfig(SharedConfig):
    data_dir: str = MISSING
    num_workers: int = 4
    pin_memory: bool = False

    masks: List[str] = MISSING
    eta: float = 0.0
    jump_length: int = 10
    jump_n_sample: int = 10


@dataclass
class OutpaintingConfig(SharedConfig):
    data_dir: str = MISSING
    num_workers: int = 4
    pin_memory: bool = False

    num_spans: int = 2
    eta: float = 0.0
    jump_length: int = 10
    jump_n_sample: int = 10
