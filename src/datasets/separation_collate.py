import torch


def collate_fn(dataset_items):
    mix = []
    targets = []
    names = []
    audio_paths = []

    for item in dataset_items:
        mix.append(item["mix"])
        names.append(item["name"])
        audio_paths.append(item["audio_path"])
        if "targets" in item:
            targets.append(item["targets"])

    batch = {
        "mix": torch.stack(mix),
        "name": names,
        "audio_path": audio_paths,
    }
    if targets:
        batch["targets"] = torch.stack(targets)
    if "video" in dataset_items[0]:
        batch["video"] = torch.stack([item["video"] for item in dataset_items])

    return batch
