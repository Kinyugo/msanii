from typing import Any, Callable, Optional

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class Lambda(nn.Module):
    def __init__(self, fn: Callable[..., Any]) -> None:
        super().__init__()

        self.fn = fn

    def forward(self, *args: Any) -> Any:
        return self.fn(*args)


def InputFn(d_in: int, d_model: int) -> nn.Sequential:
    return nn.Sequential(
        Rearrange("b c f t -> b f c t"),
        nn.Conv2d(d_in, d_model, kernel_size=1),
    )


def OutputFn(d_out: int, d_model: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(d_model, d_out, kernel_size=1),
        Rearrange("b f c t -> b c f t"),
    )


class TimestepEmbedding(nn.Module):
    def __init__(self, d_timestep: int, d_hidden_factor: int) -> None:
        super().__init__()

        self.d_timestep = d_timestep

        self.fn = nn.Sequential(
            nn.Linear(d_timestep, d_timestep * d_hidden_factor),
            nn.GELU(),
            nn.Linear(d_timestep * d_hidden_factor, d_timestep),
            Rearrange("b c -> b c () ()"),
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, x: Tensor) -> Tensor:
        half_d_timestep = self.d_timestep // 2
        emb = np.log(10000) / (half_d_timestep - 1)
        emb = torch.exp(torch.arange(half_d_timestep, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1).to(self.dtype)
        emb = self.fn(emb)

        return emb


class ResidualBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden_factor: int,
        dilation: int = 1,
        d_timestep: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.in_fn = nn.Sequential(
            nn.InstanceNorm2d(d_in),
            nn.Conv2d(d_in, d_out, kernel_size=3, dilation=dilation, padding="same"),
        )
        self.out_fn = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(d_out, d_out * d_hidden_factor, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_out * d_hidden_factor, d_out, kernel_size=1),
        )
        self.timestep_fn = (
            nn.Conv2d(d_timestep, d_out, kernel_size=1) if d_timestep else None
        )
        self.residual_fn = (
            nn.Conv2d(d_in, d_out, kernel_size=1) if d_in != d_out else nn.Identity()
        )

    def forward(self, x: Tensor, timestep_embed: Optional[Tensor] = None) -> Tensor:
        x_hidden = self.in_fn(x)
        if self.timestep_fn:
            x_hidden = x_hidden + self.timestep_fn(timestep_embed)

        return self.out_fn(x_hidden) + self.residual_fn(x)


class LinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()

        self.scale = 1 / np.sqrt(d_model // n_heads)

        self.in_fn = nn.Sequential(
            nn.InstanceNorm2d(d_model),
            nn.Conv2d(d_model, d_model * 3, kernel_size=1),
            Lambda(lambda x: torch.chunk(x, chunks=3, dim=1)),
            Lambda(
                lambda x: [
                    rearrange(t, "b (h f) c t -> (b c) h f t", h=n_heads) for t in x
                ]
            ),
        )
        self.out_fn = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.in_fn(x)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / v.shape[-1]

        context = torch.einsum("b h k l, b h v l -> b h k v", k, v)

        out = torch.einsum("b h d v, b h d l -> b h v l", context, q)
        out = rearrange(out, "(b c) h f t -> b (h f) c t", c=x.shape[-2])

        return self.out_fn(out) + x


class Block(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden_factor: int,
        dilation: int = 1,
        d_timestep: Optional[int] = None,
        n_heads: int = 8,
        has_attention: bool = True,
    ) -> None:
        super().__init__()

        self.residual_fn = ResidualBlock(
            d_in, d_out, d_hidden_factor, dilation, d_timestep
        )
        self.attention_fn = (
            LinearAttention(d_out, n_heads) if has_attention else nn.Identity()
        )

    def forward(self, x: Tensor, timestep_embed: Optional[Tensor] = None) -> Tensor:
        return self.attention_fn(self.residual_fn(x, timestep_embed))


class MiddleBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden_factor: int,
        n_heads: int = 8,
        dilation: int = 1,
        d_timestep: int = None,
    ) -> None:
        super().__init__()

        self.in_residual_fn = ResidualBlock(
            d_in, d_out, d_hidden_factor, dilation, d_timestep
        )
        self.attention_fn = LinearAttention(d_out, n_heads)
        self.out_residual_fn = ResidualBlock(
            d_in, d_out, d_hidden_factor, dilation, d_timestep
        )

    def forward(self, x: Tensor, timestep_embed: Optional[Tensor] = None) -> Tensor:
        x_hidden = self.in_residual_fn(x, timestep_embed)
        x_hidden = self.attention_fn(x_hidden)

        return self.out_residual_fn(x_hidden, timestep_embed)


def Downsample(d_model: int) -> nn.Conv2d:
    return nn.Conv2d(
        d_model, d_model, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
    )


def Upsample(d_model: int) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        d_model, d_model, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
    )
