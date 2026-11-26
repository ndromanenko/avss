import torch
import torch.nn.functional as F

from src.loss.pit_sisnr_loss import PITSISNRLoss


class DubleLoss(PITSISNRLoss):
    def __init__(self, lambda_video=0.1, temperature=0.1):
        super().__init__()
        self.lambda_video = lambda_video
        self.temperature = temperature

    def forward(self, predictions, targets, video_features, audio_features, **batch):
        audio_loss = super().forward(predictions, targets)["loss"]
        batch_size, speakers, _ = targets.shape
        frames = video_features.shape[-1]

        video_features = F.normalize(video_features.float(), dim=1)
        audio_features = F.normalize(audio_features.float(), dim=1)

        video_features = video_features.reshape(
            batch_size, speakers, -1, frames
        ).permute(0, 3, 1, 2)
        audio_features = audio_features.reshape(
            batch_size, speakers, -1, frames
        ).permute(0, 3, 1, 2)

        similarities = torch.matmul(
            video_features, audio_features.transpose(-1, -2)
        ).mean(dim=1)
        labels = torch.arange(speakers, device=predictions.device).repeat(batch_size)
        ce_loss = F.cross_entropy(similarities.flatten(0, 1) / self.temperature, labels)

        return {
            "loss": audio_loss + self.lambda_video * ce_loss,
            "audio_loss": audio_loss,
            "ce_loss": ce_loss,
        }
