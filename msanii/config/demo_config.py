from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DemoConfig:
    ckpt_path: str = MISSING
    device: str = "cpu"
    dtype: str = "float32"
