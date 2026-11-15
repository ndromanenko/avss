import torch
from torchmetrics.functional.audio import (
    permutation_invariant_training,
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
    perceptual_evaluation_speech_quality, 
    short_time_objective_intelligibility
)

from functools import partial 

from src.metrics.base_metric import BaseMetric


def get_pit(predictions, targets, metric_func):
    return permutation_invariant_training(
        predictions, targets,
        metric_func=metric_func,
        mode="speaker-wise",
        eval_func="max",
    )


class SISNRMetric(BaseMetric):
    @torch.no_grad()
    def __call__(self, predictions, targets, **batch):
        scores, _ = get_pit(predictions, targets, scale_invariant_signal_noise_ratio)
        res = scores.mean().item()
        return res


class SISNRiMetric(BaseMetric):
    @torch.no_grad()
    def __call__(self, predictions, targets, mix, **batch):
        scores, _ = get_pit(predictions, targets, scale_invariant_signal_noise_ratio)
        mixture = mix.unsqueeze(1).expand_as(targets)
        input_scores = scale_invariant_signal_noise_ratio(mixture, targets)
        res = (scores - input_scores.mean(dim=1)).mean().item()
        return res


class SDRiMetric(BaseMetric):
    @torch.no_grad()
    def __call__(self, predictions, targets, mix, **batch):
        scores, _ = get_pit(predictions, targets, signal_distortion_ratio)
        mixture = mix.unsqueeze(1).expand_as(targets)
        input_scores = signal_distortion_ratio(mixture, targets)
        res = (scores - input_scores.mean(dim=1)).mean().item()
        return res


class PESQMetric(BaseMetric):
    def __init__(self, fs=16000, mode="wb", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_func = partial(perceptual_evaluation_speech_quality, fs=fs, mode=mode)

    @torch.no_grad()
    def __call__(self, predictions, targets, **batch):
        scores, _ = get_pit(predictions, targets, self.metric_func)
        res = scores.mean().item()
        return res


class STOIMetric(BaseMetric):
    def __init__(self, fs=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_func = partial(short_time_objective_intelligibility, fs=fs)

    @torch.no_grad()
    def __call__(self, predictions, targets, **batch):
        scores, _ = get_pit(predictions, targets, self.metric_func)
        res = scores.mean().item()
        return res