import torch
from torchmetrics.functional.audio import (
    permutation_invariant_training,
    scale_invariant_signal_noise_ratio,
)

from src.metrics.base_metric import BaseMetric


def get_sisnr(predictions, targets):
    return permutation_invariant_training(
        predictions, targets,
        metric_func=scale_invariant_signal_noise_ratio,
        mode="speaker-wise",
        eval_func="max",
    )


class SISNRMetric(BaseMetric):
    @torch.no_grad()
    def __call__(self, predictions, targets, **batch):
        scores, _ = get_sisnr(predictions, targets)
        res = scores.mean().item()
        return res


class SISNRiMetric(BaseMetric):
    @torch.no_grad()
    def __call__(self, predictions, targets, mix, **batch):
        scores, _ = get_sisnr(predictions, targets)
        mixture = mix.unsqueeze(1).expand_as(targets)
        input_scores = scale_invariant_signal_noise_ratio(mixture, targets)
        res = (scores - input_scores.mean(dim=1)).mean().item()
        return res
