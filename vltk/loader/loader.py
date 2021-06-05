from copy import deepcopy
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from vltk.loader.visndataset import VisionDataset
from vltk.loader.visnlangdataset import VisionLanguageDataset


def collate(
    columns: List[Dict[str, torch.Tensor]], pad: bool = True, img_first=False
) -> Dict[str, torch.Tensor]:
    batch = {}
    columns = sorted(columns, key=lambda x: len(x), reverse=True)
    keys = deepcopy(list(columns[0].keys()))
    for k in keys:
        try:

            batch[k] = torch.stack([i.get(k) for i in columns if i is not None])
        except Exception:

            batch[k] = [i.get(k, "") for i in columns if i is not None]

    return batch


class VisionLanguageLoader(DataLoader):
    def __init__(self, config, is_train=False, **kwargs):
        if not is_train:
            num_workers = 0
        else:
            num_workers = config.num_workers
        shuffle = config.shuffle if is_train else 0
        drop_last = config.drop_last
        pin_memory = config.pin_memory
        # init dataset
        dataset = VisionLanguageDataset(config=config, is_train=is_train, **kwargs)
        # init loader
        super().__init__(
            dataset=dataset,
            collate_fn=lambda x: collate(x, img_first=config.img_first),
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            batch_size=dataset.batch_size,
        )


class VisionLoader(DataLoader):
    def __init__(self, config, is_train=False, **kwargs):
        if not is_train:
            num_workers = 0
        else:
            num_workers = config.num_workers
        shuffle = config.shuffle if is_train else 0
        drop_last = config.drop_last
        pin_memory = config.pin_memory
        # init dataset
        dataset = VisionDataset(config=config, is_train=is_train, **kwargs)
        # init loader
        super().__init__(
            dataset=dataset,
            collate_fn=lambda x: collate(x, img_first=config.img_first),
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            batch_size=dataset.batch_size,
        )
