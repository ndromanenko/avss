import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional.audio import pit_permutate
from tqdm.auto import tqdm

from src.metrics.separation import get_sisnr
from src.trainer.base_trainer import BaseTrainer


class SeparationTrainer(BaseTrainer):
    def process_batch(self, batch, metrics):
        batch = self.move_batch_to_device(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        device_type = torch.device(self.device).type
        use_amp = self.cfg_trainer.use_amp and device_type == "cuda"
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
            if self.cfg_trainer.get("duble_loss", False):
                outputs = self.model(batch["mix"], batch["video"], batch["targets"])
                batch.update(outputs)
            elif "video" in batch:
                batch["predictions"] = self.model(batch["mix"], batch["video"])
            else:
                batch["predictions"] = self.model(batch["mix"])
            batch["predictions"] = batch["predictions"].float()

        losses = self.criterion(**batch)
        batch.update(losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step()

        batch_size = batch["mix"].shape[0]
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=batch_size)
        for metric in metric_funcs:
            metrics.update(metric.name, metric(**batch), n=batch_size)
        return batch

    def _train_epoch(self, epoch):
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, total=self.epoch_len, desc="train")
        ):
            batch = self.process_batch(batch, self.train_metrics)
            self.train_metrics.update("grad_norm", self._get_grad_norm())

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.writer.add_scalar(
                    "learning rate", self.optimizer.param_groups[0]["lr"]
                )
                self._log_scalars(self.train_metrics)
            if batch_idx + 1 >= self.epoch_len:
                break

        logs = self.train_metrics.result()
        for part, dataloader in self.evaluation_dataloaders.items():
            values = self._evaluation_epoch(epoch, part, dataloader)
            logs.update({f"{part}_{name}": value for name, value in values.items()})

        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            self.lr_scheduler.step(logs[self.cfg_trainer.lr_scheduler_metric])
        return logs

    def _resume_checkpoint(self, resume_path):
        super()._resume_checkpoint(resume_path)
        for parameter, state in self.optimizer.state.items():
            if "exp_avg" in state:
                state["exp_avg"] = state["exp_avg"].reshape_as(parameter)
                state["exp_avg_sq"] = state["exp_avg_sq"].reshape_as(parameter)

    def _from_pretrained(self, pretrained_path):
        if self.config.model.get("video_features") is None:
            return super()._from_pretrained(pretrained_path)

        self.logger.info(f"Loading model weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode != "train":
            self.log_audio(**batch)

    @torch.no_grad()
    def log_audio(self, mix, targets, predictions, name, **batch):
        _, permutations = get_sisnr(predictions, targets)
        predictions = pit_permutate(predictions, permutations)

        for i in range(min(self.cfg_trainer.audio_examples, mix.shape[0])):
            self.writer.add_text(f"name_{i}", name[i])
            self.writer.add_audio(f"mix_{i}", mix[i], self.config.sample_rate)
            for speaker in range(2):
                self.writer.add_audio(
                    f"target_s{speaker + 1}_{i}",
                    targets[i, speaker],
                    self.config.sample_rate,
                )
                self.writer.add_audio(
                    f"prediction_s{speaker + 1}_{i}",
                    predictions[i, speaker],
                    self.config.sample_rate,
                )
