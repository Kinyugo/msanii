from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ..modules import (
    Block,
    Downsample,
    InputFn,
    MiddleBlock,
    OutputFn,
    TimestepEmbedding,
    Upsample,
)


class UNet(nn.Module):
    def __init__(
        self,
        d_freq: int = 128,
        d_base: int = 256,
        d_hidden_factor: int = 4,
        d_multipliers: List[int] = [1, 1, 1, 1, 1, 1, 1],
        d_timestep: int = 128,
        dilations: List[int] = [1, 1, 1, 1, 1, 1, 1],
        n_heads: int = 8,
        has_attention: List[bool] = [False, False, False, False, False, True, True],
        has_resampling: List[bool] = [True, True, True, True, True, True, False],
        n_block_layers: List[int] = [2, 2, 2, 2, 2, 2, 2],
    ) -> None:
        super().__init__()

        self.d_freq = d_freq
        self.d_base = d_base
        self.d_hidden_factor = d_hidden_factor
        self.d_multipliers = d_multipliers
        self.d_timestep = d_timestep
        self.dilations = dilations
        self.n_heads = n_heads
        self.has_attention = has_attention
        self.has_resampling = has_resampling
        self.n_block_layers = n_block_layers

        self.input_fn = InputFn(self.d_freq, self.d_base)
        self.output_fn = OutputFn(self.d_freq, self.d_base)
        self.timestep_embedding = TimestepEmbedding(
            self.d_timestep, self.d_hidden_factor
        )

        self.encoder_blocks = self.__make_encoder_blocks()
        self.middle_block = MiddleBlock(
            self.d_base * self.d_multipliers[-1],
            self.d_base * self.d_multipliers[-1],
            self.d_hidden_factor,
            self.n_heads,
            dilation=1,
            d_timestep=self.d_timestep,
        )
        self.decoder_blocks = self.__make_decoder_blocks()

    def forward(self, x: Tensor, timestep: Tensor) -> Tensor:
        x_embed = self.input_fn(x)
        timestep_embed = self.timestep_embedding(timestep)

        x_hidden, enc_hiddens = self.__encode(x_embed, timestep_embed)
        x_hidden = self.middle_block(x_hidden, timestep_embed)
        x_hidden = self.__decode(x_hidden, enc_hiddens, timestep_embed)

        return self.output_fn(x_hidden)

    def __encode(
        self, x_hidden: Tensor, timestep_embed: Tensor
    ) -> Tuple[Tensor, List[Tensor]]:
        enc_hiddens = []
        for block in self.encoder_blocks:
            if isinstance(block, Block):
                x_hidden = block(x_hidden, timestep_embed)
                enc_hiddens.append(x_hidden)
            else:
                x_hidden = block(x_hidden)

        return x_hidden, enc_hiddens

    def __decode(
        self, x_hidden: Tensor, enc_hiddens: List[Tensor], timestep_embed: Tensor
    ) -> Tensor:
        for block in self.decoder_blocks:
            if isinstance(block, Block):
                x_hidden = torch.cat((x_hidden, enc_hiddens.pop()), dim=1)
                x_hidden = block(x_hidden, timestep_embed)
            else:
                x_hidden = block(x_hidden)

        return x_hidden

    def __make_encoder_blocks(self) -> nn.ModuleList:
        blocks = nn.ModuleList()

        for idx, (d_in, d_out) in enumerate(self.__make_d_pairs()):
            # Append layers for the current blocks
            for _ in range(self.n_block_layers[idx]):
                blocks.append(
                    self.__make_block(
                        d_in, d_out, self.dilations[idx], self.has_attention[idx]
                    )
                )
                d_in = d_out

            # Append downsampling block
            if self.has_resampling[idx]:
                blocks.append(Downsample(d_out))

        return blocks

    def __make_decoder_blocks(self) -> nn.ModuleList:
        blocks = nn.ModuleList()

        # Append blocks in reverse order
        for idx, (d_out, d_in) in enumerate(self.__make_d_pairs()[::-1]):
            # Append upsampling blocks
            if self.has_resampling[::-1][idx]:
                blocks.append(Upsample(d_in))

            # Append layers for the current block
            inner_blocks = nn.ModuleList()
            for _ in range(self.n_block_layers[::-1][idx]):
                inner_blocks.append(
                    self.__make_block(
                        d_in * 2,
                        d_out,
                        self.dilations[::-1][idx],
                        self.has_attention[::-1][idx],
                    )
                )
                d_out = d_in

            # Append layers in reversed order
            blocks.extend(inner_blocks[::-1])

        return blocks

    def __make_d_pairs(self) -> nn.ModuleList:
        dims = np.multiply(self.d_multipliers + self.d_multipliers[-1:], self.d_base)
        return list(zip(dims[:-1], dims[1:]))

    def __make_block(
        self, d_in: int, d_out: int, dilation: int, has_attention: bool
    ) -> Block:
        return Block(
            d_in,
            d_out,
            self.d_hidden_factor,
            dilation,
            self.d_timestep,
            self.n_heads,
            has_attention,
        )
