from copy import deepcopy
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from vltk.loader.datatset import UniversalDataset


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


class UniversalLoader(DataLoader):
    def __init__(
        self,
        names,
        config,
        label_dict,
        textsetdict,
        imagesetdict,
        annotationdict=None,
    ):
        splits = set()
        for v in textsetdict.values():
            splits = splits.union(set(v.keys()))
        if "train" not in splits or "pretrain" not in splits:
            num_workers = 0
        else:
            num_workers = config.num_workers
        shuffle = config.shuffle
        shuffle = shuffle if set(splits).union(config.train_aliases) else 0
        drop_last = config.drop_last
        pin_memory = config.pin_memory
        # init dataset
        dataset = UniversalDataset(
            names=names,
            config=config,
            label_dict=label_dict,
            textsetdict=textsetdict,
            imagesetdict=imagesetdict,
            annotationdict=annotationdict,
        )
        # init loader
        super().__init__(
            dataset=dataset,
            collate_fn=lambda x: collate(
                x, pad=config.pad_collate, img_first=config.img_first
            ),
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            batch_size=dataset.batch_size,
        )
