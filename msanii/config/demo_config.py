from dataclasses import dataclass
from typing import Optional


@dataclass
class DemoConfig:
    pipeline_ckpt_path: str
    device: str = "cpu"
