from src.datasets.separation_dataset import SeparationDataset


class CustomDirDataset(SeparationDataset):
    def __init__(self, dir, use_video=False, limit=None):
        super().__init__(dir, part=None, limit=limit, use_video=use_video)