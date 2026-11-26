from pathlib import Path

import hydra
import soundfile as sf
import torch
from hydra.utils import instantiate
from tqdm.auto import tqdm


def load_audio(path):
    audio, _ = sf.read(path, dtype="float32")
    return torch.from_numpy(audio)


def load_example(mix_path, target_dir, pred_paths):
    name = mix_path.name
    return {
        "mix": load_audio(mix_path),
        "targets": torch.stack(
            [load_audio(target_dir / s / name) for s in ("s1", "s2")]
        ),
        "predictions": torch.stack([load_audio(p) for p in pred_paths]),
    }


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metrics")
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = [instantiate(m) for m in config.metrics.inference]
    target_dir, pred_dir = Path(config.target_dir), Path(config.pred_dir)

    pairs = []
    for mix_path in sorted((target_dir / "mix").iterdir()):
        pred_paths = [pred_dir / "s1" / mix_path.name, pred_dir / "s2" / mix_path.name]
        if not all(p.exists() for p in pred_paths):
            print(f"Нет предсказаний для {mix_path.stem}")
            continue
        pairs.append((mix_path, pred_paths))

    sums = {m.name: 0 for m in metrics}
    count = 0
    for start in tqdm(range(0, len(pairs), config.batch_size)):
        examples = [
            load_example(mix, target_dir, preds)
            for mix, preds in pairs[start : start + config.batch_size]
        ]

        by_length = {}
        for ex in examples:
            by_length.setdefault(ex["mix"].shape[-1], []).append(ex)

        for same in by_length.values():
            batch = {k: torch.stack([ex[k] for ex in same]).to(device) for k in same[0]}
            n = len(same)
            for m in metrics:
                sums[m.name] += m(**batch) * n
            count += n

    for name, value in sums.items():
        print(f"{name} - {value / count}")


if __name__ == "__main__":
    main()
