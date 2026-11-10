import torch
from torch import Tensor, nn
from torchmetrics.functional.audio import permutation_invariant_training, scale_invariant_signal_noise_ratio


class PITSISNRLoss(nn.Module):
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        scores, _ = permutation_invariant_training(
            predictions,
            targets,
            metric_func=scale_invariant_signal_noise_ratio,
            mode="speaker-wise",
            eval_func="max",
        )
        loss = -scores.mean()

        return {"loss": loss}
