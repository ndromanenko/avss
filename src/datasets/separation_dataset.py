from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.utils.io_utils import ROOT_PATH


class SeparationDataset(Dataset):
    def __init__(self, dir, part=None, limit=None):
        self.audio_dir = ROOT_PATH / dir / "audio"
        if part is not None:
            self.audio_dir = self.audio_dir / part

        self.files = []
        for path in sorted((self.audio_dir / "mix").iterdir()):
            if path.is_file() and path.suffix.lower() in (".wav", ".flac", ".mp3"):
                self.files.append(path)
        self.files = self.files[:limit]
        self.has_targets = (self.audio_dir / "s1").exists()

    def __len__(self):
        return len(self.files)

    def load_audio(self, path):
        audio, non = sf.read(path, dtype="float32")
        return torch.from_numpy(audio)

    def __getitem__(self, index):
        path = self.files[index]
        item = {
            "mix": self.load_audio(path),
            "name": path.name,
            "audio_path": str(path),
        }
        if self.has_targets:
            s1 = self.load_audio(self.audio_dir / "s1" / path.name)
            s2 = self.load_audio(self.audio_dir / "s2" / path.name)
            item["targets"] = torch.stack([s1, s2])
        return item
