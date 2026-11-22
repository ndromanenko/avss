import torch
from torch import nn
import torch.nn.functional as F

from src.model.video_encoder import VideoEncoder


class Encoder(nn.Module):
    def __init__(self, n_filters, kernel_size, stride):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)

        return x


class IntraChunkBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, input_size)
        self.gln = nn.GroupNorm(num_groups=1, num_channels=input_size, eps=1e-8)

    def forward(self, x):
        B, N, K, S = x.shape
        x_new = x.permute(0, 3, 2, 1).reshape(B * S, K, N)
        x_new, _ = self.bi_lstm(x_new)
        x_new = self.fc(x_new)
        x_new = x_new.reshape(B, S, K, N).permute(0, 3, 2, 1)
        x_new = self.gln(x_new)
        return x + x_new


class InterChunkBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.bi_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, input_size)
        self.gln = nn.GroupNorm(num_groups=1, num_channels=input_size, eps=1e-8)

    def forward(self, x):
        B, N, K, S = x.shape
        x_new = x.permute(0, 2, 3, 1).reshape(B * K, S, N)
        x_new, _ = self.bi_lstm(x_new)
        x_new = self.fc(x_new)
        x_new = x_new.reshape(B, K, S, N).permute(0, 3, 1, 2)
        x_new = self.gln(x_new)
        return x + x_new


class DPRNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.intra_chunk = IntraChunkBlock(input_size, hidden_size, num_layers)
        self.inter_chunk = InterChunkBlock(input_size, hidden_size, num_layers)

    def forward(self, x):
        x = self.intra_chunk(x)
        x = self.inter_chunk(x)
        return x


class ProcessingBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_blocks, chunk_size, num_speakers, video_features=None, use_concat=False):
        super().__init__()
        self.input_size = input_size 
        self.chunk_size = chunk_size
        self.num_speakers = num_speakers
        self.use_concat = use_concat
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(DPRNNBlock(input_size, hidden_size, num_layers))

        self.prelu = nn.PReLU()
        self.conv_out = nn.Conv1d(in_channels=input_size, out_channels=num_speakers * input_size, kernel_size=1)
        self.sigma = nn.Sigmoid()
        if video_features is not None:
            if use_concat:
                self.video_projection = nn.Sequential(
                    nn.Linear(input_size + num_speakers * video_features, input_size),
                    nn.ReLU(),
                    nn.Linear(input_size, input_size),
                )
            else:
                self.video_projection = nn.Linear(num_speakers * video_features, input_size, bias=False)
                nn.init.zeros_(self.video_projection.weight)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        key = prefix + "video_projection.weight"
        if key in state_dict and state_dict[key].ndim == 3:
            state_dict[key] = state_dict[key].squeeze(-1)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def _segmentation(self, x, chunk_size):
        step_size = chunk_size // 2
        rest = (-x.size(-1)) % step_size

        left_pad = step_size + rest // 2
        right_pad = step_size + rest - rest // 2

        x = F.pad(x, (left_pad, right_pad))

        return x.unfold(dimension=-1, size=chunk_size, step=chunk_size // 2,).transpose(-1, -2)

    def _overlap_add(self, x, chunk_size, original_length):
        B, N, K, S = x.shape

        step_size = chunk_size // 2
        rest = (-original_length) % step_size

        left_pad = step_size + rest // 2

        x = x.reshape(B, N * K, S)

        padded_length = (S - 1) * step_size + K

        x = F.fold(x, output_size=(1, padded_length), kernel_size=(1, K), stride=(1, step_size))

        x = x.squeeze(2)

        return x[..., left_pad:left_pad + original_length]

    def forward(self, x, video=None):
        B, _, L = x.shape
        if video is not None:
            if self.use_concat:
                video = F.interpolate(video, size=L, mode="linear", align_corners=False)
                x = torch.cat([x, video.to(x.dtype)], dim=1)
                x = self.video_projection(x.transpose(1, 2)).transpose(1, 2)
            else:
                video = self.video_projection(video.transpose(1, 2)).transpose(1, 2)
                video = F.interpolate(video, size=L, mode="linear", align_corners=False)
                x = x + video.to(x.dtype)

        x_seg = self._segmentation(x, self.chunk_size)
        x_block = x_seg
        for block in self.blocks:
            x_block = block(x_block)

        x_init = self._overlap_add(x_block, self.chunk_size, L)

        masks = self.prelu(x_init)
        masks = self.conv_out(masks)
        masks = self.sigma(masks)

        masks = masks.reshape(B, self.num_speakers, self.input_size, L)

        return masks 

class Decoder(nn.Module):
    def __init__(self, in_size, kernel_size, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels=in_size, out_channels=1, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        return x


class DualPathRNN(nn.Module):
    def __init__(self, n_filters, num_speakers, hidden_size, encoder_kernel_size, num_blocks, num_layers, chunk_size, video_features=None, video_encoder=None, use_concat=False):
        super().__init__()

        self.stride = encoder_kernel_size // 2
        self.kernel_size = encoder_kernel_size

        self.encoder = Encoder(n_filters, encoder_kernel_size, encoder_kernel_size // 2)
        self.separation = ProcessingBlock(n_filters, hidden_size, num_layers, num_blocks, chunk_size, num_speakers, video_features, use_concat)
        self.decoder = Decoder(n_filters, encoder_kernel_size, encoder_kernel_size // 2)
        self.video_encoder = video_encoder
        if self.video_encoder is None and video_features is not None:
            self.video_encoder = VideoEncoder(video_features)


    def freeze_audio(self):
        self.requires_grad_(False)
        self.video_encoder.requires_grad_(True)
        self.separation.video_projection.requires_grad_(True)

    def forward(self, x, video=None, return_video_features=False):
        
        length = x.shape[-1]
        pad = (self.stride - (length - self.kernel_size) % self.stride) % self.stride
        x = F.pad(x, (0, pad))
        
        w = self.encoder(x)
        video_features = None
        if video is not None:
            video_features = self.video_encoder(video)
        masks = self.separation(w, video_features)
        s = masks * w.unsqueeze(1)

        B, C, _, _ = s.shape
        s = s.flatten(0, 1)
        x = self.decoder(s).reshape(B, C, -1)

        predictions = x[..., :length]
        if return_video_features:
            return predictions, video_features
        return predictions

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