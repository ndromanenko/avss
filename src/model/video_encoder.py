from torch import nn


class VideoEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.video_conv = nn.Conv3d(
            1, 16, kernel_size=(5, 5, 5), stride=(1, 4, 4), padding=2
        )

        self.frame_encoder = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, features),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(features, features, kernel_size=5, padding=2),
            nn.ReLU(),
        )

    def forward(self, video):
        batch_size, speakers, frames, height, width = video.shape
        x = (
            video.reshape(batch_size * speakers, 1, frames, height, width).float()
            / 255.0
        )
        x = self.video_conv(x)
        x = x.transpose(1, 2).flatten(0, 1)
        x = self.frame_encoder(x)
        x = x.reshape(batch_size * speakers, frames, -1).transpose(1, 2)
        x = self.temporal_encoder(x)
        return x.reshape(batch_size, -1, frames)
