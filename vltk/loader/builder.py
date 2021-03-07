import json
from collections import defaultdict

from vltk import SPLITALIASES, VDATA, VLDATA
from vltk.abc.imageset import Imagesets
from vltk.abc.textset import Textsets
from vltk.loader.loader import VisionLanguageLoader, VisionLoader

_textsets = Textsets()
_imagesets = Imagesets()


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
        imagesets = out_dict["imagesets"]
        answer_to_id = out_dict["answers"]
        object_to_id = out_dict["objects"]

        if train:
            train_loader = VisionLanguageLoader(
                config,
                textsetdict=train,
                imagesetdict=imagesets,
                annotationdict=annos,
                answer_to_id=answer_to_id,
                object_to_id=object_to_id,
                is_train=False,
            )
        if test:
            eval_loader = VisionLanguageLoader(
                config,
                textsetdict=train,
                imagesetdict=imagesets,
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
                imagesetdict=train,
                annotationdict=annotations,
                object_to_id=object_to_id,
                is_train=True,
            )
        if test:
            eval_loader = VisionLoader(
                config,
                imagesetdict=test,
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
    load_textsets = defaultdict(set)
    load_imagesets = defaultdict(set)
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
        if ds in _imagesets.avail():
            all_img = True
            load_imagesets[ds].update(splits)
        if ds in _textsets.avail():
            all_vl = True
            load_textsets[ds].update(splits)
    for ds, split in train:
        train_ds[ds].update(split_handler(split))
    for ds, split in test:
        eval_ds[ds].update(split_handler(split))

    assert not (all_vl and all_img), "cannot specify mixture of VL and Vision datasets"
    datasets_type = VDATA if all_img else VLDATA
    to_load = load_imagesets if all_img else load_textsets
    return train_ds, eval_ds, to_load, datasets_type


def load_vl(to_load, train_ds, eval_ds, config):
    loaded_eval = defaultdict(dict)  # will be datasetk
    loaded_train = defaultdict(dict)  # will be datasetk
    loaded_imagesets = defaultdict(dict)
    loaded_annotations = defaultdict(dict)
    answer_to_id = {}
    object_to_id = {}
    answer_id = 0
    object_id = 0
    for name in sorted(set(to_load.keys())):
        splits = split_handler(to_load[name])  # list looks like ['trainval', 'dev']
        for split in splits:
            # add textset first
            textset = _textsets.get(name).from_config(config.data, splits=split)[split]
            for l in sorted(textset.labels):
                if l not in answer_to_id:
                    answer_to_id[l] = answer_id
                    answer_id += 1
            if name in eval_ds and split in split_handler(eval_ds[name]):
                loaded_eval[name][split] = textset
            if name in train_ds and split in split_handler(train_ds[name]):
                loaded_train[name][split] = textset
            print(f"Added Textset {name}: {split}")
            # now add imageset
            is_name, is_split = zip(*textset.data_info[split].items())
            is_name = is_name[0]
            is_split = is_split[0][0]
            # first check to see if we want annotations
            if config.annotations and is_name not in loaded_annotations:
                # function in imageset that: given datadir + optional split + optional
                # extractor feats + annotation bool will return
                # the desired path
                is_annotations = _imagesets.get(is_name).load(
                    config.datadir, annotation=True
                )
                loaded_annotations[is_name] = is_annotations
                for l in sorted(textset.labels):
                    if l not in object_to_id:
                        object_to_id[l] = object_id
                        object_id += 1
            if loaded_imagesets[is_name][is_split]:
                continue
            if config.extractor is not None:
                # this is if we want to get pre-computed features
                extractor = config.extractor
                is_data = _imagesets.get(is_name).load(
                    config.datadir, split=split, extractor=extractor
                )
            else:
                # this is if we want to get raw features (in the form {id: raw file})
                is_data = _imagesets.get(is_name).load(config.datadir, split=split)
            loaded_imagesets[is_name][is_split] = is_data
            print(f"Added Imageset {is_name}: {is_split}")

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
        "imagesets": loaded_imagesets,
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
        annotations = _imagesets.get(name).load(config.datadirs[-1], annotations=True)
        loaded_annotations[name] = annotations
        for split in splits:
            imgids2pathes = _imagesets.get(name).load_imgid2path(
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
            print(f"Added Imageset {name}: {split}")
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


def init_v_loader():
    pass


def init_vl_loader():
    pass


def init_loaders():
    pass
