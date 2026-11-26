from hydra.utils import instantiate

from src.datasets.separation_collate import collate_fn
from src.utils.init_utils import set_worker_seed


def get_dataloaders(config):
    dataloaders = {}
    for part in config.datasets:
        dataset = instantiate(config.datasets[part])
        dataloaders[part] = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            shuffle=part == "train",
            drop_last=part == "train",
            worker_init_fn=set_worker_seed,
        )
    return dataloaders
