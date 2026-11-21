from uuid import uuid4

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.separation_data_utils import get_dataloaders
from src.trainer.separation_trainer import SeparationTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.io_utils import ROOT_PATH


@hydra.main(version_base=None, config_path="src/configs", config_name="conv_tasnet")
def main(config):
    set_random_seed(config.trainer.seed)
    logger = setup_saving_and_logging(config)

    if config.trainer.resume_from is None:
        config.writer.run_id = uuid4().hex
        save_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        OmegaConf.save(config, save_dir / "config.yaml")

    project_config = OmegaConf.to_container(config, resolve=True)
    writer = instantiate(config.writer, logger, project_config, _convert_="all")

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    dataloaders = get_dataloaders(config)
    model = instantiate(config.model).to(device)
    audio_encoder_checkpoint = config.trainer.get("audio_encoder_checkpoint")
    if audio_encoder_checkpoint is not None:
        model.load_audio_encoder(audio_encoder_checkpoint)
    if config.trainer.get("freeze_audio", False):
        model.freeze_audio()
    logger.info(model)
    loss_function = instantiate(config.loss_function).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in metrics:
        for metric_config in config.metrics[metric_type]:
            metrics[metric_type].append(instantiate(metric_config))

    trainable_params = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_params.append(parameter)
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    trainer = SeparationTrainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        text_encoder=None,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=config.trainer.epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=instantiate(config.transforms.batch_transforms),
        skip_oom=config.trainer.skip_oom,
    )

    try:
        trainer.train()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
