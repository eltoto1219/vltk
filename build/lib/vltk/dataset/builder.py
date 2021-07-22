import json
import os
from collections import defaultdict

import vltk
from vltk.adapters import Adapters
from vltk.dataset.loader import VisionLanguageLoader, VisionLoader

_adapters = Adapters()


def download(config, datadir, name, _adapters, seen_download):
    path = os.path.join(datadir, name)
    if os.path.exists(path) and not config.redownload:
        return seen_download
    if name in seen_download:
        return seen_download
    if name not in seen_download:
        seen_download.add(name)
    _adapters.get(name).download(datadir)
    return seen_download


def extract(
    config, datadir, name, _adapters, extracted, extractor=None, annotations=False
):
    if extractor is not None:
        if extractor in extracted:
            return extracted[extractor]
    if name in extracted:
        return extracted[name]

    if extractor is not None:
        exists = os.path.exists(os.path.join(datadir, name, extractor))
        if not exists or config.reextract:
            is_data = _adapters.get(config.extractor).extract(
                datadir, dataset_name=name
            )
            extracted[config.extractor] = is_data
            return extracted
        else:
            return extracted
    else:
        if annotations:
            exists = os.path.exists(os.path.join(datadir, name, "annotations.arrow"))
            if not exists:
                exists = os.path.exists(
                    os.path.join(datadir, name, "annotations", "annotations.arrow")
                )
            if not exists or config.reextract:
                is_annotations = _adapters.get(name)
                # is_annotations = is_annotations.extract(datadir)
                # is_annotations = is_annotations.extract(datadir)
                try:
                    is_annotations = is_annotations.extract(datadir)
                except TypeError as e:
                    if "object is not iterable" not in e:
                        print(f"Warning: No Annotations for {name}.")
                    else:
                        raise Exception(e)
                extracted[name] = is_annotations
                return extracted
            else:
                return extracted
        else:
            exists = False
            for split in vltk.SPLITALIASES:
                temp = os.path.join(datadir, name, f"{split}.arrow")
                if os.path.exists(temp):
                    exists = True
            if not exists or config.reextract:
                visnlangdataset = _adapters.get(name).extract(datadir)
                extracted[name] = visnlangdataset
                return extracted
            else:
                return extracted


def init_datasets(config):
    train_loader = None
    eval_loader = None
    metadata_ids = None
    assert (
        config.lang.ignore_id < 0
    ), f"ignore id: {config.lang.ignore_id} must be negative"
    train_ds, eval_ds, to_load, datasets_type = parse_datasets(config)
    if datasets_type == vltk.VLDATA:
        out_dict = load_vl(to_load, train_ds, eval_ds, config)
        metadata_ids = out_dict["metadata_ids"]
        train = out_dict["train"]
        evalutation = out_dict["eval"]
        annos = out_dict["annotations"]
        visndatasetadapters = out_dict["visndatasetadapters"]

        if train:
            train_loader = VisionLanguageLoader(
                config,
                visnlangdatasetadapterdict=train,
                visndatasetadapterdict=visndatasetadapters,
                annotationdict=annos,
                metadata_ids=metadata_ids,
                is_train=True,
            )

        if evalutation:
            eval_loader = VisionLanguageLoader(
                config,
                visnlangdatasetadapterdict=evalutation,
                visndatasetadapterdict=visndatasetadapters,
                annotationdict=annos,
                metadata_ids=metadata_ids,
                is_train=False,
            )
    if datasets_type == vltk.VDATA:

        out_dict = load_v(to_load, train_ds, eval_ds, config)
        # if is_train is false
        train = out_dict["train"]
        evalutation = out_dict["eval"]
        metadata_ids = out_dict["metadata_ids"]
        annotations = out_dict["annotations"]

        if train:
            train_loader = VisionLoader(
                config,
                visndatasetadapterdict=train,
                annotationdict={k: v for k, v in annotations.items() if k in train_ds},
                metadata_ids=metadata_ids,
                is_train=True,
            )
        if evalutation:
            eval_loader = VisionLoader(
                config,
                visndatasetadapterdict=evalutation,
                annotationdict={k: v for k, v in annotations.items() if k in eval_ds},
                metadata_ids=metadata_ids,
                is_train=False,
            )

    # loaders = {
    # "train": train_loader if train_loader is not None else None,
    # "eval": eval_loader if eval_loader is not None else None,
    # }
    # loaders = [(k, v) for k, v in loaders.items()]
    # any_train = any(map(lambda x: x == "train", [k for k in loaders]))
    # loaders = sorted(loaders, key=lambda x: x[0], reverse=True)
    train_loader = train_loader if train_loader is not None else None
    eval_loader = eval_loader if eval_loader is not None else None
    if train_loader is not None and hasattr(train_loader.dataset, "tokenizer"):
        train_loader.tokenizer = train_loader.dataset.tokenizer
    if eval_loader is not None and hasattr(eval_loader.dataset, "tokenizer"):
        eval_loader.tokenizer = eval_loader.dataset.tokenizer

    return train_loader, eval_loader


def parse_datasets(config):
    load_visnlangdatasets = defaultdict(set)
    load_visndatasets = defaultdict(set)
    train_ds = defaultdict(set)
    eval_ds = defaultdict(set)
    train = config.train_datasets
    train = train if train is not None else []
    test = config.eval_datasets
    test = test if test is not None else []

    assert train or test, "Must specify dataset in config to instatiate"
    if train and isinstance(train[0], str):
        train = [train]
    if test and isinstance(test[0], str):
        test = [test]
    total = train + test
    all_img = False
    all_vl = False
    for pair in total:
        ds, split = pair[0], pair[1]
        ds = ds.lower()
        split = split.lower()
        splits = split_handler(split)
        # TODO: will need to change this
        if _adapters.is_visnlang(ds):
            all_vl = True
            load_visnlangdatasets[ds].update(splits)
        if _adapters.is_visn(ds):
            load_visndatasets[ds].update(splits)
            all_img = True
    for ds, split in train:
        train_ds[ds].update(split_handler(split))
    for ds, split in test:
        eval_ds[ds].update(split_handler(split))

    assert not (all_vl and all_img), "cannot specify mixture of VL and Vision datasets"
    datasets_type = vltk.VDATA if all_img else vltk.VLDATA
    to_load = load_visndatasets if all_img else load_visnlangdatasets
    return train_ds, eval_ds, to_load, datasets_type


def load_vl(to_load, train_ds, eval_ds, config):
    datadir = config.datadir
    loaded_eval = defaultdict(dict)  # will be datasetk
    loaded_train = defaultdict(dict)  # will be datasetk
    loaded_visndatasetadapters = defaultdict(dict)
    loaded_annotations = defaultdict(dict)
    metadata_ids = {}
    metadata_idxs = {}
    seen_download = set()
    extracted = {}
    for name in sorted(set(to_load.keys())):
        # -DOWNLOAD HERE
        seen_download = download(config, datadir, name, _adapters, seen_download)
        extracted = extract(config, datadir, name, _adapters, extracted)

        splits = split_handler(to_load[name])  # list looks like ['trainval', 'dev']
        for split in sorted(splits):
            # add visnlangdatasetadapter first
            if name not in extracted:
                visnlangdatasetadapter = _adapters.get(name).load(
                    datadir, split=split, config=config
                )
            else:
                visnlangdatasetadapter = extracted["name"]["split"]
            dataset_metadata = visnlangdatasetadapter.get_metadata_counters()
            for meta in dataset_metadata:
                if meta not in metadata_ids:
                    metadata_ids[meta] = {"": config.lang.ignore_id}
                    metadata_idxs[meta] = 0
                for l in sorted(dataset_metadata[meta].keys()):
                    if l not in metadata_ids[meta]:
                        metadata_ids[meta][l] = metadata_idxs[meta]
                        metadata_idxs[meta] += 1

            if name in eval_ds and split in split_handler(eval_ds[name]):
                loaded_eval[name][split] = visnlangdatasetadapter
            if name in train_ds and split in split_handler(train_ds[name]):
                loaded_train[name][split] = visnlangdatasetadapter
            print(f"Added VisnLangDataset {name}: {split}")
            # now add visndatasetadapter
            is_name, is_split = zip(*visnlangdatasetadapter.data_info[split].items())
            is_name = is_name[0]
            is_split = is_split[0][0]
            # first check to see if we want annotations
            if not config.ignore_annotations and is_name not in loaded_annotations:

                # -DOWNLOAD HERE
                seen_download = download(
                    config, datadir, is_name, _adapters, seen_download
                )
                extracted = extract(
                    config, datadir, name, _adapters, extracted, annotations=True
                )

                if name not in extracted:
                    try:
                        is_annotations = _adapters.get(is_name).load(
                            datadir, config=config
                        )
                    except Exception:
                        print(f"Warning: No Annotations for {is_name}")
                        is_annotations = _adapters.get(is_name)
                else:
                    is_annotations = extracted[name]
                loaded_annotations[is_name] = is_annotations

                try:
                    dataset_metadata = is_annotations.get_metadata_counters()
                    for meta in dataset_metadata:
                        if meta not in metadata_ids:
                            metadata_ids[meta] = {"": config.lang.ignore_id}
                            metadata_idxs[meta] = 0
                        for l in sorted(dataset_metadata[meta].keys()):
                            if l not in metadata_ids[meta]:
                                metadata_ids[meta][l] = metadata_idxs[meta]
                                metadata_idxs[meta] += 1
                except Exception:
                    pass

            if (
                is_name in loaded_visndatasetadapters[is_name]
                and is_split in loaded_visndatasetadapters[is_name]
            ):
                continue
            if config.extractor is not None:
                # this is if we want to get pre-computed features

                extracted = extract(
                    config, datadir, name, _adapters, extractor=config.extractor
                )

                if config.extractor not in extracted:
                    is_data = _adapters.get(config.extractor).load(
                        datadir, split=is_split, dataset_name=is_name, config=config
                    )
                else:
                    is_data = extracted[config.extractor]
            else:
                # this is if we want to get raw features (in the form {id: raw file})
                is_data = _adapters.get(is_name).load_imgid2path(
                    config.datadir, split=split, config=config
                )
                # print(is_data)
            if (
                is_name in loaded_visndatasetadapters
                and is_split in loaded_visndatasetadapters[is_name]
            ):
                pass
            else:
                loaded_visndatasetadapters[is_name][is_split] = is_data
                print(f"Added VisnDataset {is_name}: {is_split}")

    if config.metadata_filedict is not None:
        metadata_filedict = config.metadata_filedict
        for k in metadata_ids:
            if k in metadata_filedict:
                metadata_ids[k] = json.load(open(metadata_filedict[k]))

    return {
        "eval": loaded_eval,
        "train": loaded_train,
        "annotations": loaded_annotations,
        "visndatasetadapters": loaded_visndatasetadapters,
        "metadata_ids": metadata_ids,
    }


# provide overlap ids at some other point
def load_v(to_load, train_ds, eval_ds, config):
    loaded_eval = defaultdict(dict)  # will be dataset
    loaded_train = defaultdict(dict)  # will be dataset
    loaded_annotations = defaultdict(dict)
    metadata_ids = {}
    metadata_idxs = {}
    seen_download = set()
    extracted = {}
    for name in sorted(set(to_load.keys())):
        # -DOWNLOAD HERE
        seen_download = download(config, config.datadir, name, _adapters, seen_download)
        extracted = extract(
            config, config.datadir, name, _adapters, extracted, annotations=True
        )
        splits = split_handler(to_load[name])  # list looks like ['trainval', 'dev']
        if name in extracted:
            annotations = extracted[name]
        else:
            annotations = _adapters.get(name).load(config.datadir, config=config)
        loaded_annotations[name] = annotations
        for split in sorted(splits):
            imgids2pathes = _adapters.get(name).load_imgid2path(config.datadir, split)

            dataset_metadata = annotations.get_metadata_counters()
            for meta in dataset_metadata:
                if meta not in metadata_ids:
                    metadata_ids[meta] = {"": config.lang.ignore_id}
                    metadata_idxs[meta] = 0
                for l in sorted(dataset_metadata[meta].keys()):
                    if l not in metadata_ids[meta]:
                        metadata_ids[meta][l] = metadata_idxs[meta]
                        metadata_idxs[meta] += 1
            if name in eval_ds and split in split in eval_ds[name]:
                loaded_eval[name][split] = imgids2pathes
            if name in train_ds and split in train_ds[name]:
                loaded_train[name][split] = imgids2pathes
            print(f"Added VisnDatasetAdapter {name}: {split}")

    if config.metadata_filedict is not None:
        metadata_filedict = config.metadata_filedict
        for k in metadata_ids:
            if k in metadata_filedict:
                metadata_ids[k] = json.load(open(metadata_filedict[k]))

    return {
        "train": loaded_train,
        "eval": loaded_eval,
        "annotations": loaded_annotations,
        "metadata_ids": metadata_ids,
    }


def split_handler(splits):
    if isinstance(splits, str):
        splits = [splits]
    unique_splits = set()
    for split in splits:
        if split == "testdev":
            unique_splits.add(split)
        else:
            for valid in vltk.SPLITALIASES:
                if valid in split:
                    unique_splits.add(valid)
    if "" in splits:
        unique_splits.add("")
    return sorted(unique_splits)
