import torch
import torch.nn.functional as F
from torch import nn

from src.model.dprnn import DualPathRNN, Encoder
from src.utils.io_utils import ROOT_PATH


class DubleLossDualPathRNN(DualPathRNN):
    def __init__(self, audio_features=256, audio_kernel_size=32, **kwargs):
        super().__init__(**kwargs)
        self.audio_target_encoder = Encoder(
            audio_features, audio_kernel_size, audio_kernel_size // 2
        )
        self.audio_target_encoder.requires_grad_(False)
        self.ce_projection = nn.Conv1d(
            kwargs["video_features"], audio_features, kernel_size=1
        )

    def load_audio_encoder(self, checkpoint_path):
        checkpoint = torch.load(
            ROOT_PATH / checkpoint_path, map_location="cpu", weights_only=False
        )
        state_dict = {}

        for name, value in checkpoint["state_dict"].items():
            if name.startswith("encoder."):
                state_dict[name.removeprefix("encoder.")] = value

        self.audio_target_encoder.load_state_dict(state_dict)

    def freeze_audio(self):
        super().freeze_audio()
        self.ce_projection.requires_grad_(True)

    def forward(self, x, video=None, targets=None):
        if targets is None:
            return super().forward(x, video)

        predictions, video_features = super().forward(
            x, video, return_video_features=True
        )

        batch_size, speakers, _ = targets.shape
        frames = video_features.shape[-1]
        video_features = video_features.reshape(batch_size * speakers, -1, frames)
        video_features = self.ce_projection(video_features)

        with torch.no_grad():
            audio_features = self.audio_target_encoder(targets.flatten(0, 1))
            audio_features = F.adaptive_avg_pool1d(audio_features, frames)

        return {
            "predictions": predictions,
            "video_features": video_features,
            "audio_features": audio_features,
        }
