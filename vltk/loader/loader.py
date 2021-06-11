from copy import deepcopy
from typing import Dict, List, Union

import torch
import vltk
from torch.utils.data import DataLoader
from vltk.loader.visndataset import VisionDataset
from vltk.loader.visnlangdataset import VisionLanguageDataset


def collate(
    columns: List[Dict[str, torch.Tensor]],
    pad: bool = True,
    img_first=False,
    all_same_keys: bool = True,
    max_spanning_cols: Union[None, set] = None,
    collate_simple: bool = True,
) -> Dict[str, torch.Tensor]:

    if all_same_keys:
        return collate_homogeneous(columns, pad, img_first)
    elif collate_simple:
        return collate_min_spanning_cols(columns, pad, img_first)
    else:
        return collate_heterogenous(columns, pad, img_first, max_spanning_cols)


def collate_homogeneous(
    columns: List[Dict[str, torch.Tensor]],
    pad: bool = True,
    img_first=False,
):
    batch = {}
    for k in columns[0].keys():
        if k == vltk.segmentation:
            batch[k] = torch.stack([i.get(k) for i in columns if i is not None])
        else:
            try:

                batch[k] = torch.stack([i.get(k) for i in columns if i is not None])
            except Exception:

                batch[k] = [i.get(k, "") for i in columns if i is not None]

    return batch


def collate_min_spanning_cols(
    columns: List[Dict[str, torch.Tensor]],
    pad: bool = True,
    img_first=False,
):
    batch = {}
    columns = sorted(columns, key=lambda x: len(x), reverse=True)
    for k in columns[0].keys():
        try:

            batch[k] = torch.stack([i.get(k) for i in columns if i is not None])
        except Exception:

            batch[k] = [i.get(k, None) for i in columns if i is not None]

    return batch


def collate_heterogenous(
    columns: List[Dict[str, torch.Tensor]],
    pad: bool = True,
    img_first=False,
    max_spanning_cols: Union[None, set] = None,
):
    raise Exception
    batch = {}
    for k in max_spanning_cols:
        try:

            batch[k] = torch.stack([i.get(k) for i in columns if i is not None])
        except Exception:

            batch[k] = [i.get(k, "") for i in columns if i is not None]

    return batch

    pass


def check_all_keys_same(config, visnlangdict=None, visndict=None, annodict=None):

    visn_keys_same = True
    visnlang_keys_same = True
    anno_keys_same = True
    tokenizer_in_visn_dataset = False
    visnlang_cols = set()
    visn_cols = set()
    anno_cols = set()
    if visnlangdict is not None:
        for dset in visnlangdict:
            for split in visnlangdict[dset]:
                adapter = visnlangdict[dset][split]
                if not visnlang_cols:
                    visnlang_cols = set(adapter.column_names)
                elif visnlang_cols != set(adapter.column_names):
                    visnlang_keys_same = False
                    visnlang_cols = visnlang_cols.union(adapter.column_names)
                else:
                    continue
    if visndict is not None:
        isadapter = True
        for dset in visndict:
            for split in visndict[dset]:
                dict_or_adapter = visndict[dset][split]
                if isinstance(dict_or_adapter, dict):
                    isadapter = False
                if not isadapter:
                    continue
                if not visn_cols:
                    visn_cols = set(dict_or_adapter.column_names)
                elif visn_cols != set(dict_or_adapter.column_names):
                    visn_keys_same = False
                    visn_cols = visn_cols.union(dict_or_adapter.column_names)
                else:
                    continue
            if not isadapter:
                continue
    if annodict is not None and not config.ignore_annotations:
        # TODO: check the case where segmentation things are different
        for adapter in annodict.values():
            if not anno_cols:
                # not so simple here, I must check that if there exists a difference, that the difference
                # only refers to a difference of segmentation datatypes
                try:
                    anno_cols = set(adapter.column_names)
                except TypeError:
                    anno_cols = set()

            try:
                adapter_cols = set(adapter.column_names)
            except Exception:
                adapter_cols = set()

            if vltk.text in adapter_cols:
                tokenizer_in_visn_dataset = True
            if anno_cols != adapter_cols:
                remaining_anno_cols = anno_cols - adapter_cols
                remaining_adapter_cols = adapter_cols - anno_cols
                remaining_cols = remaining_anno_cols.union(remaining_adapter_cols)
                if remaining_cols == set([vltk.polygons, vltk.RLE]):
                    continue
                anno_keys_same = False
                anno_cols = visnlang_cols.union(adapter.column_names)
            replace_keys = anno_cols.intersection(visnlang_cols)
            replace_keys.remove(vltk.imgid)
            anno_cols = anno_cols.union(set(map(lambda x: "v" + x, replace_keys)))
        if vltk.polygons in anno_cols or vltk.RLE in anno_cols:
            try:
                anno_cols.remove(vltk.polygons)
            except KeyError:
                pass
            try:
                anno_cols.remove(vltk.RLE)
            except KeyError:
                pass
            if not config.ignore_segmentation:
                anno_cols.add(vltk.segmentation)

    max_spanning_cols = visnlang_cols.union(visn_cols).union(anno_cols)
    print(f"Max spanning column names for each batch: {max_spanning_cols}")
    # here we will add all the feature names that could be added while we process each example
    all_same_keys = visnlang_keys_same and visn_keys_same and anno_keys_same
    if not all_same_keys and not config.collate_simple:
        print(
            "NOTICE: feature types differ between specificed datasets, so the batch dictionary will\
                                use a combination of the dataset names as keys which will point to their respective\
                                batch dictionaries."
        )
    return all_same_keys, max_spanning_cols, tokenizer_in_visn_dataset, replace_keys


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

        visnlangdict = kwargs.get("visnlangdatasetadapterdict", None)
        annodict = kwargs.get("annotationdict", None)
        visndict = kwargs.get("visndatasetadapterdict", None)
        (
            all_same_keys,
            max_spanning_cols,
            tokenizer_in_visn_dataset,
            replace_keys,
        ) = check_all_keys_same(config, visnlangdict, visndict, annodict)
        dataset = VisionLanguageDataset(
            config=config,
            is_train=is_train,
            all_same_keys=all_same_keys,
            tokenizer_in_visn_dataset=tokenizer_in_visn_dataset,
            **kwargs,
        )
        # init loader
        super().__init__(
            dataset=dataset,
            collate_fn=lambda x: collate(
                x,
                img_first=config.img_first,
                all_same_keys=all_same_keys,
                max_spanning_cols=max_spanning_cols,
                collate_simple=config.collate_simple,
            ),
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
        visnlangdict = kwargs.get("visnlangdatasetadapterdict", None)
        annodict = kwargs.get("annotationdict", None)
        visndict = kwargs.get("visndatasetadapterdict", None)

        (
            all_same_keys,
            max_spanning_cols,
            tokenizer_in_visn_dataset,
        ) = check_all_keys_same(config, visnlangdict, visndict, annodict)
        dataset = VisionDataset(
            config=config,
            is_train=is_train,
            all_same_keys=all_same_keys,
            tokenizer_in_visn_dataset=tokenizer_in_visn_dataset,
            **kwargs,
        )

        # init loader
        super().__init__(
            dataset=dataset,
            collate_fn=lambda x: collate(
                x,
                img_first=config.img_first,
                all_same_keys=all_same_keys,
                max_spanning_cols=max_spanning_cols,
                collate_simple=config.collate_simple,
            ),
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            batch_size=dataset.batch_size,
        )
