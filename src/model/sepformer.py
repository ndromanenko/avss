import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10_000):
        super().__init__()

        positions = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            -math.log(10_000) * torch.arange(0, embed_dim, 2) / embed_dim
        )

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].to(
            device=x.device,
            dtype=x.dtype,
        )


class Encoder(nn.Module):
    def __init__(self, n_filters, kernel_size, stride):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffw = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):

        z = self.norm1(x)
        attention, _ = self.mha(z, z, z, need_weights=False)
        x = x + attention
        z = self.norm2(x)
        x = x + self.ffw(z)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, max_len):
        super().__init__()

        self.pos_enc = PositionEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerLayer(
                    embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim
                )
            )

    def forward(self, x):
        residual = x
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x)

        return x + residual


class IntraChunkBlock(nn.Module):
    def __init__(self, input_size, num_heads, ff_dim, num_layers, max_len):
        super().__init__()

        self.transformer = TransformerBlock(
            embed_dim=input_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            max_len=max_len,
        )

    def forward(self, x):
        B, N, K, S = x.shape

        x = x.permute(0, 3, 2, 1)
        x = x.reshape(B * S, K, N)

        x = self.transformer(x)

        x = x.reshape(B, S, K, N)
        x = x.permute(0, 3, 2, 1)

        return x


class InterChunkBlock(nn.Module):
    def __init__(self, input_size, num_heads, ff_dim, num_layers, max_len):
        super().__init__()

        self.transformer = TransformerBlock(
            embed_dim=input_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            max_len=max_len,
        )

    def forward(self, x):
        B, N, K, S = x.shape

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * K, S, N)

        x = self.transformer(x)

        x = x.reshape(B, K, S, N)
        x = x.permute(0, 3, 1, 2)

        return x


class SepFormerBlock(nn.Module):
    def __init__(self, input_size, num_heads, ff_dim, num_layers, max_len):
        super().__init__()
        self.intra_chunk = IntraChunkBlock(
            input_size, num_heads, ff_dim, num_layers, max_len
        )
        self.inter_chunk = InterChunkBlock(
            input_size, num_heads, ff_dim, num_layers, max_len
        )

    def forward(self, x):
        x = self.intra_chunk(x)
        x = self.inter_chunk(x)
        return x


class ProcessingBlock(nn.Module):
    def __init__(
        self,
        input_size,
        ff_dim,
        num_layers,
        num_blocks,
        num_heads,
        chunk_size,
        num_speakers,
        max_len,
        mask_ff_dim,
    ):
        super().__init__()
        self.input_size = input_size
        self.chunk_size = chunk_size
        self.num_speakers = num_speakers
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                SepFormerBlock(
                    input_size=input_size,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    num_layers=num_layers,
                    max_len=max_len,
                )
            )

        self.prelu = nn.PReLU()
        self.linear = nn.Linear(input_size, num_speakers * input_size)

        self.ffw = nn.Sequential(
            nn.Linear(input_size, mask_ff_dim),
            nn.ReLU(),
            nn.Linear(mask_ff_dim, input_size),
            nn.ReLU(),
        )

        self.input_norm = nn.GroupNorm(1, input_size)
        self.input_linear = nn.Conv1d(input_size, input_size, 1)

    def _segmentation(self, x, chunk_size):
        step_size = chunk_size // 2
        rest = (-x.size(-1)) % step_size

        left_pad = step_size + rest // 2
        right_pad = step_size + rest - rest // 2

        x = F.pad(x, (left_pad, right_pad))

        return x.unfold(
            dimension=-1,
            size=chunk_size,
            step=chunk_size // 2,
        ).transpose(-1, -2)

    def _overlap_add(self, x, chunk_size, original_length):
        B, N, K, S = x.shape

        step_size = chunk_size // 2
        rest = (-original_length) % step_size

        left_pad = step_size + rest // 2

        x = x.reshape(B, N * K, S)

        padded_length = (S - 1) * step_size + K

        x = F.fold(
            x, output_size=(1, padded_length), kernel_size=(1, K), stride=(1, step_size)
        )

        x = x.squeeze(2)

        return x[..., left_pad : left_pad + original_length]

    def forward(self, x):
        B, _, L = x.shape

        x = self.input_norm(x)
        x = self.input_linear(x)

        x_seg = self._segmentation(x, self.chunk_size)
        x_block = x_seg
        for block in self.blocks:
            x_block = block(x_block)

        B, N, K, S = x_block.shape

        masks = self.prelu(x_block)

        masks = masks.permute(0, 2, 3, 1)
        masks = self.linear(masks)
        masks = masks.permute(0, 3, 1, 2)

        masks = self._overlap_add(masks, self.chunk_size, L)

        masks = masks.reshape(B, self.num_speakers, N, L)

        masks = masks.permute(0, 1, 3, 2)
        masks = self.ffw(masks)
        masks = masks.permute(0, 1, 3, 2)

        return masks


class Decoder(nn.Module):
    def __init__(self, in_size, kernel_size, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels=in_size, out_channels=1, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SepFormer(nn.Module):
    def __init__(
        self,
        n_filters,
        num_speakers,
        ff_dim,
        encoder_kernel_size,
        num_blocks,
        num_layers,
        num_heads,
        chunk_size,
        max_len,
        mask_ff_dim,
    ):
        super().__init__()

        self.stride = encoder_kernel_size // 2
        self.kernel_size = encoder_kernel_size

        self.encoder = Encoder(n_filters, encoder_kernel_size, encoder_kernel_size // 2)
        self.separation = ProcessingBlock(
            input_size=n_filters,
            ff_dim=ff_dim,
            num_layers=num_layers,
            num_blocks=num_blocks,
            num_heads=num_heads,
            chunk_size=chunk_size,
            num_speakers=num_speakers,
            max_len=max_len,
            mask_ff_dim=mask_ff_dim,
        )
        self.decoder = Decoder(n_filters, encoder_kernel_size, encoder_kernel_size // 2)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):

        length = x.shape[-1]
        pad = (self.stride - (length - self.kernel_size) % self.stride) % self.stride
        x = F.pad(x, (0, pad))

        w = self.encoder(x)
        masks = self.separation(w)
        s = masks * w.unsqueeze(1)

        B, C, _, _ = s.shape
        s = s.flatten(0, 1)
        x = self.decoder(s).reshape(B, C, -1)

        return x[..., :length]

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
