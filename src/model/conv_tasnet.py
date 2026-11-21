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

class Decoder(nn.Module):
    def __init__(self, in_size, kernel_size, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels=in_size, out_channels=1, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv1DBlock(nn.Module):
    def __init__(self, b_size, h_size, kernel_size, dilation, skip_size):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=b_size, out_channels=h_size, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.gln1 = nn.GroupNorm(num_groups=1, num_channels=h_size, eps=1e-8)
        self.d_conv = nn.Conv1d(in_channels=h_size, out_channels=h_size, kernel_size=kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2, groups=h_size)
        self.prelu2 = nn.PReLU()
        self.gln2 = nn.GroupNorm(num_groups=1, num_channels=h_size, eps=1e-8)
        self.residual_conv = nn.Conv1d(in_channels=h_size, out_channels=b_size, kernel_size=1)
        self.skip_connetion_conv = nn.Conv1d(in_channels=h_size, out_channels=skip_size, kernel_size=1)

    def forward(self, x):
        x_new = self.conv1(x)
        x_new = self.prelu1(x_new)
        x_new = self.gln1(x_new)
        x_new = self.d_conv(x_new)
        x_new = self.prelu2(x_new)
        x_new = self.gln2(x_new)
        residual_branch = x + self.residual_conv(x_new)
        skip_conn_branch = self.skip_connetion_conv(x_new)


        return residual_branch, skip_conn_branch

class Separation(nn.Module):
    def __init__(self, in_size, b_size, h_size, skip_size, kernel_size, dilation_factor, repeat, num_speakers, video_features=None):
        super().__init__()

        self.in_size = in_size
        self.num_speakers = num_speakers
        self.gln1 = nn.GroupNorm(num_groups=1, num_channels=in_size, eps=1e-8)
        self.conv_in = nn.Conv1d(in_channels=in_size, out_channels=b_size, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.conv_out = nn.Conv1d(in_channels=skip_size, out_channels=num_speakers * in_size, kernel_size=1)
        self.sigma = nn.Sigmoid()
        self.blocks = nn.ModuleList()
        self.video_block = dilation_factor if repeat > 1 else 0
        if video_features is not None:
            self.video_projection = nn.Conv1d(num_speakers * video_features, b_size, 1, bias=False)
            nn.init.zeros_(self.video_projection.weight)

        for _ in range(repeat):
            for d in range(dilation_factor):
                self.blocks.append(Conv1DBlock(b_size, h_size, kernel_size, 2**d, skip_size))

    def forward(self, x, video=None):
        batch_size, _, T = x.shape
        x_new = self.gln1(x)
        x_new = self.conv_in(x_new)

        skip_sum = 0
        x_block = x_new
        for index, block in enumerate(self.blocks):
            if video is not None and index == self.video_block:
                video = self.video_projection(video)
                video = F.interpolate(
                    video, size=T, mode="linear", align_corners=False
                )
                x_block = x_block + video.to(x_block.dtype)
            residual_branch, skip_conn_branch = block(x_block)
            x_block = residual_branch
            skip_sum += skip_conn_branch

        masks = self.prelu1(skip_sum)
        masks = self.conv_out(masks)
        masks = self.sigma(masks)

        masks = masks.reshape(batch_size, self.num_speakers, self.in_size, T)

        return masks


class ConvTaSNet(nn.Module):
    def __init__(self, n_filters, n_speakers, dilation_factor, repeat, b_size, h_size, skip_size, encoder_kernel_size, tcn_kernel_size, video_features=None, video_encoder=None):
        super().__init__()

        self.stride = encoder_kernel_size // 2
        self.kernel_size = encoder_kernel_size

        self.encoder = Encoder(n_filters, encoder_kernel_size, encoder_kernel_size // 2)
        self.sep = Separation(
            in_size=n_filters, 
            b_size=b_size, 
            h_size=h_size, 
            skip_size=skip_size,
            kernel_size=tcn_kernel_size, 
            dilation_factor=dilation_factor, 
            repeat=repeat, 
            num_speakers=n_speakers,
            video_features=video_features,
        )
        self.decoder = Decoder(n_filters, encoder_kernel_size, encoder_kernel_size // 2)
        self.video_encoder = video_encoder
        if self.video_encoder is None and video_features is not None:
            self.video_encoder = VideoEncoder(video_features)


    def freeze_audio(self):
        self.requires_grad_(False)
        self.video_encoder.requires_grad_(True)
        self.sep.video_projection.requires_grad_(True)

    def forward(self, x, video=None):
        
        length = x.shape[-1]
        pad = (self.stride - (length - self.kernel_size) % self.stride) % self.stride
        x = F.pad(x, (0, pad))
        
        w = self.encoder(x)
        video_features = None
        if video is not None:
            video_features = self.video_encoder(video)
        masks = self.sep(w, video_features)
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
