import json
from collections import defaultdict

from vltk import SPLITALIASES, VDATA, VLDATA
from vltk.adapters import Adapters
from vltk.loader.loader import VisionLanguageLoader, VisionLoader

_adapters = Adapters()


def init_datasets(config):
    train_loader = None
    eval_loader = None
    answer_to_id = None
    object_to_id = None
    train_ds, eval_ds, to_load, datasets_type = parse_datasets(config)
    if datasets_type == VLDATA:
        out_dict = load_vl(to_load, train_ds, eval_ds, config)
        train = out_dict["train"]
        test = out_dict["eval"]
        annos = out_dict["annotations"]
        visndatasetadapters = out_dict["visndatasetadapters"]
        answer_to_id = out_dict["answers"]
        object_to_id = out_dict["objects"]

        if train:
            train_loader = VisionLanguageLoader(
                config,
                visnlangdatasetadapterdict=train,
                visndatasetadapterdict=visndatasetadapters,
                annotationdict=annos,
                answer_to_id=answer_to_id,
                object_to_id=object_to_id,
                is_train=True,
            )
        if test:
            eval_loader = VisionLanguageLoader(
                config,
                visnlangdatasetadapterdict=train,
                visndatasetadapterdict=visndatasetadapters,
                annotationdict=annos,
                answer_to_id=answer_to_id,
                object_to_id=object_to_id,
                is_train=False,
            )
    if datasets_type == VDATA:

        out_dict = load_v(to_load, train_ds, eval_ds, config)
        # if is_train is false
        train = out_dict["train"]
        test = out_dict["eval"]
        object_to_id = out_dict["objects"]
        annotations = out_dict["annotations"]

        if train:
            train_loader = VisionLoader(
                config,
                visndatasetadapterdict=train,
                annotationdict=annotations,
                object_to_id=object_to_id,
                is_train=True,
            )
        if test:
            eval_loader = VisionLoader(
                config,
                visndatasetadapterdict=test,
                annotationdict=annotations,
                object_to_id=object_to_id,
                is_train=False,
            )

    loaders = {
        "train": train_loader if train_loader is not None else None,
        "eval": eval_loader if eval_loader is not None else None,
    }
    loaders = [(k, v) for k, v in loaders.items()]
    any_train = any(map(lambda x: x == "train", [k for k in loaders]))
    loaders = sorted(loaders, key=lambda x: x[0], reverse=True)

    return loaders, any_train, answer_to_id, object_to_id


def parse_datasets(config):
    load_visnlangdatasetadapters = defaultdict(set)
    load_visndatasetadapters = defaultdict(set)
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
        if ds in _adapters.avail():
            all_img = True
            load_visndatasetadapters[ds].update(splits)
        if ds in _adapters.avail():
            all_vl = True
            load_visnlangdatasetadapters[ds].update(splits)
    for ds, split in train:
        train_ds[ds].update(split_handler(split))
    for ds, split in test:
        eval_ds[ds].update(split_handler(split))

    assert not (all_vl and all_img), "cannot specify mixture of VL and Vision datasets"
    datasets_type = VDATA if all_img else VLDATA
    to_load = load_visndatasetadapters if all_img else load_visnlangdatasetadapters
    return train_ds, eval_ds, to_load, datasets_type


def load_vl(to_load, train_ds, eval_ds, config):
    loaded_eval = defaultdict(dict)  # will be datasetk
    loaded_train = defaultdict(dict)  # will be datasetk
    loaded_visndatasetadapters = defaultdict(dict)
    loaded_annotations = defaultdict(dict)
    answer_to_id = {}
    object_to_id = {}
    answer_id = 0
    object_id = 0
    for name in sorted(set(to_load.keys())):
        splits = split_handler(to_load[name])  # list looks like ['trainval', 'dev']
        for split in splits:
            # add visnlangdatasetadapter first
            visnlangdatasetadapter = _adapters.get(name).load(config, splits=split)
            for l in sorted(visnlangdatasetadapter.labels):
                if l not in answer_to_id:
                    answer_to_id[l] = answer_id
                    answer_id += 1
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
            if config.annotations and is_name not in loaded_annotations:
                # function in visndatasetadapter that: given datadir + optional split + optional
                # extractor feats + annotation bool will return
                # the desired path
                is_annotations = _adapters.get(is_name).load(
                    config.datadir, annotations=True
                )
                loaded_annotations[is_name] = is_annotations
                for l in sorted(visnlangdatasetadapter.labels):
                    if l not in object_to_id:
                        object_to_id[l] = object_id
                        object_id += 1
            if (
                is_name in loaded_visndatasetadapters[is_name]
                and is_split in loaded_visndatasetadapters[is_name]
            ):
                continue
            if config.extractor is not None:
                # this is if we want to get pre-computed features
                extractor = config.extractor
                is_data = _adapters.get(is_name).load(
                    config.datadir, extractor=extractor, split=is_split
                )
            else:
                # this is if we want to get raw features (in the form {id: raw file})
                is_data = _adapters.get(is_name).load_imgid2path(
                    config.datadir, split=split
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

        answer_file = config.labels
        objects_file = config.objects_file
        if answer_file is not None or "":
            answer_to_id = json.load(open(answer_file))
        if objects_file is not None or "":
            object_to_id = json.load(open(objects_file))

    return {
        "eval": loaded_eval,
        "train": loaded_train,
        "annotations": loaded_annotations,
        "visndatasetadapters": loaded_visndatasetadapters,
        "answers": answer_to_id,
        "objects": object_to_id,
    }


# provide overlap ids at some other point
def load_v(to_load, train_ds, eval_ds, config):
    loaded_eval = defaultdict(dict)  # will be datasetk
    loaded_train = defaultdict(dict)  # will be datasetk
    loaded_annotations = defaultdict(dict)
    object_to_id = {}
    object_id = 0
    for name in sorted(set(to_load.keys())):
        splits = split_handler(to_load[name])  # list looks like ['trainval', 'dev']
        annotations = _adapters.get(name).load(config.datadirs[-1], annotations=True)
        loaded_annotations[name] = annotations
        for split in splits:
            imgids2pathes = _visndatasetadapters.get(name).load_imgid2path(
                config.datadirs[-1], split
            )
            for l in sorted(annotations.labels):
                if l not in object_to_id:
                    object_to_id[l] = object_id
                    object_id += 1
            if name in eval_ds and split in split in eval_ds[name]:
                loaded_eval[name][split] = imgids2pathes
            if name in train_ds and split in train_ds[name]:
                loaded_train[name][split] = imgids2pathes
            print(f"Added VisnDatasetAdapter {name}: {split}")
    if config.objects_file is not None:
        object_to_id = json.load(config.objects_file)

    return {
        "train": loaded_train,
        "eval": loaded_eval,
        "annotations": loaded_annotations,
        "objects": object_to_id,
    }


def split_handler(splits):
    if isinstance(splits, str):
        splits = [splits]
    unique_splits = set()
    for split in splits:
        if split == "testdev":
            unique_splits.add(split)
        else:
            for valid in SPLITALIASES:
                if valid in split:
                    unique_splits.add(valid)
    return sorted(unique_splits)
