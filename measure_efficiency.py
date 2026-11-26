import os
import tempfile
import time

import hydra
import torch
from hydra.utils import instantiate
from torch.utils.flop_counter import FlopCounterMode

from src.datasets.custom_dir_dataset import CustomDirDataset
from src.datasets.separation_collate import collate_fn


@hydra.main(version_base=None, config_path="src/configs", config_name="efficiency")
def main(config):
    ckpt = torch.load(config.checkpoint, map_location="cuda", weights_only=False)
    model = instantiate(ckpt["config"]["model"]).cuda().eval()
    model.load_state_dict(ckpt["state_dict"])

    batch = collate_fn([CustomDirDataset(config.data_dir, use_video=True)[0]])
    mix, video = batch["mix"].cuda(), batch["video"].cuda()

    params = sum(p.numel() for p in model.parameters())

    with tempfile.NamedTemporaryFile(suffix=".pth") as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / 2**20

    with torch.inference_mode():
        with FlopCounterMode(display=False) as counter:
            model(mix, video)

        for _ in range(config.n_warmup):
            model(mix, video)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        model(mix, video)
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 2**20

        times = []
        for _ in range(config.n_runs):
            start = time.perf_counter()
            model(mix, video)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"params: {params / 1e6:.3f} M")
    print(f"size on disk: {size_mb:.2f} MB")
    print(f"MACs: {counter.get_total_flops() / 2 / 1e9:.3f} G")
    print(f"peak GPU memory: {peak_mb:.1f} MB")
    print(f"time per batch: {1000 * sum(times) / len(times):.2f} ms")


if __name__ == "__main__":
    main()
