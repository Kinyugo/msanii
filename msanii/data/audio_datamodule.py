from typing import Optional

import lightning as L
from torch.utils.data import DataLoader

from .audio_dataset import AudioDataset


class AudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 44_100,
        num_frames: Optional[int] = None,
        load_random_slice: bool = False,
        normalize_amplitude: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        shuffle: bool = True,
    ) -> None:

        super().__init__()

        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.load_random_slice = load_random_slice
        self.normalize_amplitude = normalize_amplitude
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

    def setup(self, stage: str = None) -> None:
        self.dataset = AudioDataset(
            self.data_dir,
            self.sample_rate,
            self.num_frames,
            self.load_random_slice,
            self.normalize_amplitude,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )
