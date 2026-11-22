import torch
from pystoi import stoi as pystoi_stoi
from torchmetrics.functional.audio import (
    permutation_invariant_training,
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
    perceptual_evaluation_speech_quality
)

from functools import partial
import multiprocessing as mp

from src.metrics.base_metric import BaseMetric


def get_pit(predictions, targets, metric_func):
    return permutation_invariant_training(
        predictions, targets,
        metric_func=metric_func,
        mode="speaker-wise",
        eval_func="max",
    )

def get_sisnr(predictions, targets):
    return get_pit(predictions, targets, scale_invariant_signal_noise_ratio)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_func = partial(signal_distortion_ratio, zero_mean=True)

    @torch.no_grad()
    def __call__(self, predictions, targets, mix, **batch):
        scores, _ = get_pit(predictions, targets, self.metric_func)
        mixture = mix.unsqueeze(1).expand_as(targets)
        input_scores = self.metric_func(mixture, targets)
        res = (scores - input_scores.mean(dim=1)).mean().item()
        return res


class PESQMetric(BaseMetric):
    def __init__(self, fs=16000, mode="wb", n_processes=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_func = partial(perceptual_evaluation_speech_quality, fs=fs, mode=mode, n_processes=n_processes)

    @torch.no_grad()
    def __call__(self, predictions, targets, **batch):
        scores, _ = get_pit(predictions, targets, self.metric_func)
        res = scores.mean().item()
        return res


class STOIMetric(BaseMetric):
    def __init__(self, fs=16000, extended=False, n_processes=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fs = fs
        self.extended = extended
        self.n_processes = n_processes
        self._pool = None

    def _map(self, args):
        if self.n_processes <= 1:
            return [pystoi_stoi(*a) for a in args]
        if self._pool is None:
            self._pool = mp.get_context("fork").Pool(self.n_processes)
        return self._pool.starmap(pystoi_stoi, args)

    def _stoi_batch(self, preds, target):
        p = preds.detach().float().cpu().numpy()
        t = target.detach().float().cpu().numpy()
        vals = self._map([(t[i], p[i], self.fs, self.extended) for i in range(len(p))])
        return torch.tensor(vals, dtype=torch.float32, device=preds.device)

    @torch.no_grad()
    def __call__(self, predictions, targets, **batch):
        scores, _ = get_pit(predictions, targets, self._stoi_batch)
        return scores.mean().item()