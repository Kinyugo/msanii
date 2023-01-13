import re
from typing import List, Tuple

import torch
from torch import Tensor


def generate_batch_audio_mask(
    mask_strs: List[str], audio: Tensor, sample_rate: int
) -> Tensor:
    """Generate audio masks for a batch of audio."""
    batch_intervals = list(map(mask_intervals_from_str, mask_strs))

    audio_mask = torch.ones_like(audio)
    for i, intervals in enumerate(batch_intervals):
        for start, end in intervals:
            start_idx = int(start * sample_rate)
            end_idx = int(end * sample_rate)
            audio_mask[i, ..., start_idx:end_idx] = 0

    return audio_mask


def mask_intervals_from_str(mask_str: str) -> List[Tuple[int, int]]:
    """Generates a list of start and end for mask intervals,
    e.g: '3-5, 6-7' gives [(3,5), (6,7)]
    """
    intervals = re.findall(r"(\d+)-(\d+)", mask_str)
    intervals = [tuple(map(int, x)) for x in intervals]
    intervals.sort()

    return intervals
