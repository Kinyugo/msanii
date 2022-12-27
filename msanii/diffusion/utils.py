from typing import Optional

import torch
from torch import Tensor


def noise_like(x: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)


def sequential_mask(like: Tensor, start: int) -> Tensor:
    length, device = like.shape[-1], like.device
    mask = torch.ones_like(like, dtype=torch.bool)
    mask[..., start:] = torch.zeros((length - start,), device=device)

    return mask
