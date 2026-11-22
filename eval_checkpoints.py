import glob
from pathlib import Path

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.separation_collate import collate_fn
from src.datasets.separation_dataset import SeparationDataset


@torch.no_grad()
def evaluate(ckpt_path, config, metrics, device):
    run_dir = Path(ckpt_path).parent

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    run_cfg = ckpt["config"]
    use_video = run_cfg.model.get("video_features") is not None

    model = instantiate(run_cfg.model).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    dataset = instantiate(config.datasets.val, use_video=use_video)
    loader = instantiate(config.dataloader, dataset=dataset, collate_fn=collate_fn, shuffle=False)

    sums = {m.name: 0 for m in metrics}
    count = 0
    for batch in tqdm(loader, desc=run_dir.name):
        mix = batch["mix"].to(device)
        targets = batch["targets"].to(device)
        if use_video:
            preds = model(mix, batch["video"].to(device))
        else:
            preds = model(mix)

        b = {"mix": mix, "targets": targets, "predictions": preds.float()}
        n = mix.shape[0]
        for m in metrics:
            sums[m.name] += m(**b) * n
        count += n

    row = {
        "run": run_dir.name,
        "model": run_cfg.model._target_.split(".")[-1],
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    }
    row.update({name: round(s / count, 4) for name, s in sums.items()})

    del model
    torch.cuda.empty_cache()
    return row


@hydra.main(version_base=None, config_path="src/configs", config_name="eval_checkpoints")
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = [instantiate(m) for m in config.metrics.inference]

    out = Path(config.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out) if out.exists() else pd.DataFrame()

    ckpt_paths = sorted(glob.glob(config.checkpoints))
    if not ckpt_paths:
        print(f"Не найдено чекпоинтов по шаблону: {config.checkpoints}")
        return

    for ckpt_path in ckpt_paths:
        run = Path(ckpt_path).parent.name
        if "run" in df and run in set(df["run"]):
            print(f"skip {run}: уже есть в {out}")
            continue

        row = evaluate(ckpt_path, config, metrics, device)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(out, index=False)
        print(pd.DataFrame([row]).to_string(index=False))

    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()