from typing import Tuple, TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SCLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()

        self.eps = eps

    def forward(self, input_specgram: Tensor, target_specgram: Tensor) -> Tensor:
        return torch.norm((target_specgram - input_specgram), p="fro") / (
            torch.norm(target_specgram, p="fro") + self.eps
        )


class LogMagnitudeLoss(nn.Module):
    def __init__(self, distance: str = "l1", eps: float = 1e-6) -> None:
        super().__init__()

        self.distance = distance
        self.eps = eps

    def forward(self, input_specgram: Tensor, target_specgram: Tensor) -> Tensor:
        input_specgram = torch.log(input_specgram + self.eps)
        target_specgram = torch.log(target_specgram + self.eps)

        if self.distance == "l1":
            return F.l1_loss(input_specgram, target_specgram)
        return F.mse_loss(input_specgram, target_specgram)


class SpectrogramLossDict(TypedDict):
    sc_loss: Tensor
    lm_loss: Tensor


class SpectrogramLoss(nn.Module):
    def __init__(
        self,
        w_sc_loss: float = 1.0,
        w_lm_loss: float = 1.0,
        eps: float = 1e-6,
        distance: str = "l1",
    ) -> None:
        super().__init__()

        self.w_sc_loss = w_sc_loss
        self.w_lm_loss = w_lm_loss

        self.sc_loss = SCLoss(eps)
        self.lm_loss = LogMagnitudeLoss(distance, eps)

    def forward(
        self, input_specgram: Tensor, target_specgram: Tensor
    ) -> Tuple[Tensor, SpectrogramLossDict]:
        sc_loss = self.w_sc_loss * self.sc_loss(input_specgram, target_specgram)
        lm_loss = self.w_lm_loss * self.lm_loss(input_specgram, target_specgram)

        total_loss = sc_loss + lm_loss
        losses = SpectrogramLossDict(sc_loss=sc_loss, lm_loss=lm_loss)

        return total_loss, losses
