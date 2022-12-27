import os
import random
from typing import Optional, Tuple

import torchaudio
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        sample_rate: int,
        num_frames: Optional[int] = None,
        load_random_slice: bool = False,
        normalize_amplitude: bool = True,
    ) -> None:
        super().__init__()

        self.data_dir = os.path.expanduser(data_dir)
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.load_random_slice = load_random_slice
        self.normalize_amplitude = normalize_amplitude

        self.filenames = os.listdir(self.data_dir)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tensor:
        filepath = os.path.join(self.data_dir, self.filenames[index])

        if self.load_random_slice:
            waveform, sample_rate = self.__load_random_audio_slice(filepath)
        else:
            waveform, sample_rate = torchaudio.load(filepath)

        waveform = self.__resample(waveform, sample_rate)
        waveform = self.__pad(waveform)
        waveform = self.__normalize_amplitude(waveform)

        return waveform.clamp(min=-1.0, max=1.0)

    def __load_random_audio_slice(self, filepath: str) -> Tuple[Tensor, int]:
        metadata = torchaudio.info(filepath)
        frames_to_load = int(
            (metadata.sample_rate / self.sample_rate) * self.num_frames
        )
        frame_offset = random.randint(0, max(0, metadata.num_frames - frames_to_load))

        waveform, sample_rate = torchaudio.load(
            filepath, num_frames=frames_to_load, frame_offset=frame_offset
        )

        return waveform, sample_rate

    def __resample(self, x: Tensor, sample_rate: int) -> Tensor:
        if sample_rate != self.sample_rate:
            return torchaudio.functional.resample(x, sample_rate, self.sample_rate)
        return x

    def __pad(self, x: Tensor) -> Tensor:
        if self.num_frames:
            return F.pad(x, (0, self.num_frames - x.shape[-1]), value=0.0)
        return x

    def __normalize_amplitude(self, x: Tensor) -> Tensor:
        if self.normalize_amplitude:
            return x / x.abs().max()
        return x
