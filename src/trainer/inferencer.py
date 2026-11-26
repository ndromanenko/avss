from pathlib import Path

import soundfile as sf
import torch
from torchmetrics.functional.audio import pit_permutate
from tqdm.auto import tqdm

from src.metrics.separation import get_sisnr
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)

        if "video" in batch:
            predictions = self.model(batch["mix"], batch["video"])
        else:
            predictions = self.model(batch["mix"])
        batch["predictions"] = predictions.float()

        if "targets" in batch:
            _, perm = get_sisnr(batch["predictions"], batch["targets"])
            batch["predictions"] = pit_permutate(batch["predictions"], perm)
            if metrics is not None:
                for met in self.metrics["inference"]:
                    metrics.update(met.name, met(**batch), n=batch["mix"].shape[0])

        if self.save_path is not None:
            self._save_predictions(batch, part)
        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()
        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        if self.save_path is not None:
            for spk in ("s1", "s2"):
                (self.save_path / part / spk).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc=part, total=len(dataloader)
            ):
                self.process_batch(batch_idx, batch, self.evaluation_metrics, part)

        has_targets = dataloader.dataset.has_targets
        return (
            self.evaluation_metrics.result()
            if has_targets and self.evaluation_metrics
            else {}
        )

    def _save_predictions(self, batch, part):
        preds = batch["predictions"].detach().cpu()

        preds = preds - preds.mean(dim=-1, keepdim=True)

        peak = preds.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        preds = 0.9 * preds / peak

        for i, name in enumerate(batch["name"]):
            for spk in range(preds.shape[1]):
                out = self.save_path / part / f"s{spk + 1}" / name
                sf.write(
                    out,
                    preds[i, spk].numpy(),
                    self.config.sample_rate,
                )
