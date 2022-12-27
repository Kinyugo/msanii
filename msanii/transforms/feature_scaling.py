from typing import Tuple

import torch
from torch import Tensor, nn


class StandardScaler(nn.Module):
    def __init__(
        self, momentum: float = 1e-3, momentum_decay: float = 1e-3, eps: float = 1e-5
    ) -> None:
        super().__init__()

        self.eps = eps
        self.momentum_decay = momentum_decay

        self.register_buffer("momentum", torch.tensor(momentum))
        self.register_buffer("running_mean", torch.tensor(0.0))
        self.register_buffer("running_var", torch.tensor(1.0))
        self.register_buffer("fitted", torch.tensor(False))

    def forward(self, x: Tensor, inverse: bool = False) -> Tensor:
        if inverse:
            return self.inverse_transform(x)
        return self.transform(x)

    def transform(self, x: Tensor) -> Tensor:
        # Compute & update statistics over the current batch in training mode
        if self.training:
            batch_mean = x.mean()
            batch_var = x.var()

            self.__update_stats(batch_mean, batch_var)

        # Use running statistics in other modes
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        return (x - batch_mean) / torch.sqrt(batch_var + self.eps)

    def inverse_transform(self, x: Tensor) -> Tensor:
        # Use running statistics to undo the standardization
        return (x * torch.sqrt(self.running_var + self.eps)) + self.running_mean

    @torch.no_grad()
    def step(self) -> None:
        self.momentum.data = self.momentum_decay * self.momentum.data

    @torch.no_grad()
    def __update_stats(self, batch_mean: Tensor, batch_var: Tensor) -> Tensor:
        # Copy batch statistics for the initial fitting
        if not self.fitted:
            self.running_mean.data = batch_mean.data
            self.running_var.data = batch_var.data
            self.fitted.data = torch.tensor(True, device=batch_mean.device)

        # Update a moving average of the statistics
        else:
            self.running_mean.data = (
                1.0 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var.data = (
                1.0 - self.momentum
            ) * self.running_var + self.momentum * batch_var


class MinMaxScaler(nn.Module):
    def __init__(
        self,
        feature_range: Tuple[float, float] = (-1.0, 1.0),
        momentum: float = 1e-3,
        momentum_decay=1e-3,
        clip: bool = True,
    ) -> None:
        super().__init__()

        self.feature_range = feature_range
        self.momentum_decay = momentum_decay
        self.clip = clip

        self.register_buffer("momentum", torch.tensor(momentum))
        self.register_buffer("running_min", torch.tensor(0.0))
        self.register_buffer("running_max", torch.tensor(1.0))
        self.register_buffer("fitted", torch.tensor(False))

    def forward(self, x: Tensor, inverse: bool = False) -> Tensor:
        if inverse:
            return self.inverse_transform(x)
        return self.transform(x)

    def transform(self, x: Tensor) -> Tensor:
        f_min, f_max = self.feature_range

        # Compute & update statistics over the current batch in training mode
        if self.training:
            batch_min = x.min()
            batch_max = x.max()

            self.__update_stats(batch_min, batch_max)

        # Use running statistics in other modes
        else:
            batch_min = self.running_min
            batch_max = self.running_max

        x_transformed = (x - batch_min) / (batch_max - batch_min)
        x_transformed = x_transformed * (f_max - f_min) + f_min

        # Ensure values are in the appropriate range
        if self.clip:
            return torch.clip(x_transformed, min=f_min, max=f_max)
        return x_transformed

    def inverse_transform(self, x: Tensor) -> Tensor:
        f_min, f_max = self.feature_range

        # Ensure values are in the appropriate range
        if self.clip:
            x = torch.clip(x, min=f_min, max=f_max)

        # Use running statistics to undo the min-max scaling
        x_transformed = (x - f_min) / (f_max - f_min)
        x_transformed = (
            x_transformed * (self.running_max - self.running_min) + self.running_min
        )

        return x_transformed

    @torch.no_grad()
    def step(self) -> None:
        self.momentum.data = self.momentum_decay * self.momentum.data

    @torch.no_grad()
    def __update_stats(self, batch_min: Tensor, batch_max: Tensor) -> Tensor:
        # Copy batch statistics for the initial fitting
        if not self.fitted:
            self.running_min.data = batch_min.data
            self.running_max.data = batch_max.data
            self.fitted.data = torch.tensor(True, device=batch_min.device)

        # Update a moving average of the statistics
        else:
            self.running_min.data = (
                1.0 - self.momentum
            ) * self.running_min + self.momentum * batch_min
            self.running_max.data = (
                1.0 - self.momentum
            ) * self.running_max + self.momentum * batch_max
