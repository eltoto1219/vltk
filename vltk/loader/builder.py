from collections import defaultdict

VLDATA = 0
VDATA = 1
LDATA = 2
valid_eval = ['testdev', 'test', 'dev', 'eval', 'val', 'validation', 'evaulation']
valid_train = ['train', 'val', 'validation', 'dev']

textset_avail = ['vqa', 'gqa']
imageset_avial = ['coco', 'vg']

test1 = ['vqa', 'trainval']
test2 = [['vqa', 'trainval'], ['gqa', 'trainval']]
test3 = ['gqa', 'dev']
test4 = [['gqa', 'test'], ['coco', 'train']]

valid_splits  = set(['testdev', 'test', 'dev', 'eval', 'val', 'validation', 'evaluation', 'train'])

#lets just do for train first
class Config:
    train_datasets = test1
    eval_datasets = test3

def init_datasets(config):
    train_ds, eval_ds, to_load, datasets_type = parse_datasets(config)
    if datasets_type == VLDATA:
        out_dict = load_vl(to_load, train_ds, eval_ds, config)
    if datasets_type == VDATA:
        out_dict = load_v(to_load, train_ds, eval_ds, config)


def parse_datasets(config):
    load_textsets = defaultdict(set)
    load_imagesets = defaultdict(set)
    train_ds = defaultdict(set)
    eval_ds = defaultdict(set)
    train = config.train_datasets
    train = train if train is not None else []
    test = config.eval_datasets
    test = test if test is not None else []
    if isinstance(train[0], str):
        train = [train]
    if isinstance(test[0], str):
        test = [test]
    total = train + test
    all_img = False
    all_ds = False
    for pair in total:
        ds, split = pair[0], pair[1]
        if ds in imageset_avial:
            all_img = True
            load_imagesets[ds].add(split)
        if ds in textset_avail:
            all_vl = True
            load_textsets[ds].add(split)
    for ds, split  in train:
        train_ds[ds].add(split)
    for ds, split  in test:
        eval_ds[ds].add(split)

    assert not (all_vl and all_img), "cannot specify mixture of VL and Vision datasets"
    datasets_type =  VDATA if all_img else VLDATA
    to_load = load_imagesets if all_img else load_textsets
    return train_ds, eval_ds, to_load, datasets_type



def load_vl(to_load, tain_ds, eval_ds, config):
    loaded_eval = defaultdict(dict) # will be datasetk
    loaded_train = defaultdict(dict) # will be datasetk
    loaded_imagesets = defaultdict(dict)
    loaded_annotations = defaultdict(dict)
    answer_to_id = {}
    object_to_id = {}
    answer_id = 0
    object_id = 0
    for name in sorted(set(to_load.keys())):
            splits = unique_splits(to_load[name]) # list looks like ['trainval', 'dev']
            for split in splits:
                #add textset first
                textset = _textsets.get(name).from_config(self.config.data, splits=split)[split]
                for l in sorted(textset.labels):
                    if l not in answer_to_id:
                        answer_to_id[l] = label_id
                        label_id += 1
                if name in eval_ds and split in unique_splits(eval_ds[name]):
                    loaded_eval[name][split] = textset
                if name in train_ds and split in unique_splits(train_ds[name]):
                    loaded_train[name][split] = textset
                print(f"Added Textset {name}: {split}")
                # now add imageset
                is_name, is_split = zip(*textset.data_info[split].items())
                is_name = is_name[0]
                is_split = is_split[0][0]
                # first check to see if we want annotations
                if self.config.annotations and is_name not in loaded_annotations:
                    #function in imageset that: given datadir + optional split + optional
                    # extractor feats + annotation bool will return
                    #the desired path
                    is_annotations = _imagesets.get(is_name).load(config.datadir, annotation=True)
                    loaded_annotations[is_name] = is_annotations
                    for l in sorted(textset.labels):
                        if l not in self.label_to_id:
                            object_to_id[l] = label_id
                            object_id += 1
                if loaded_imagesets[is_name][is_split]:
                    continue
                if self.config.extractor is not None:
                    #this is if we want to get pre-computed features
                    extractor = self.config.extractor
                    is_data = _imagesets.get(is_name).load(config.datadir, split=split, extractor=extractor)
                else:
                    #this is if we want to get raw features (in the form {id: raw file})
                    is_data = _imagesets.get(is_name).load(config.datadir, split=split)
                loaded_imagesets[is_name][is_split] = is_data
                print(f"Added Imageset {is_name}: {is_split}")
    return {
        'eval_datasets': loaded_eval,
        'train_datasets': loaded_train,
        'annotations': loaded_annotations,
        'imagesets': loaded_imagesets,
        'answers': answer_to_id,
        'objects': object_to_id,
    }

# provide overlap ids at some other point
def load_v(to_load, tain_ds, eval_ds, config):
    loaded_eval = defaultdict(dict) # will be datasetk
    loaded_train = defaultdict(dict) # will be datasetk
    loaded_annotations = defaultdict(dict)
    object_to_id = {}
    object_id = 0
    for name in sorted(set(to_load.keys())):
        splits = unique_splits(to_load[name]) # list looks like ['trainval', 'dev']
        annotations = _imagesets.get(name).load(self.config.datadir, annotations=True)
        loaded_annotations[name] = annotations
        for split in splits:
            #add textset first
            imageset = _imagesets.get(name).load(self.config.datadir, splits=split)
            for l in sorted(annotations.labels):
                if l not in object_to_id:
                    object_to_id[l] = label_id
                    object_id += 1
            if name in eval_ds and split in split in eval_ds[name]:
                loaded_eval[name][split] = imageset
            if name in train_ds and split in train_ds[name]:
                loaded_train[name][split] = imageset
            print(f"Added Textset {name}: {split}")
    return {
        'train': loaded_train,
        'eval': loaded_eval,
        'annotations': loaded_annotations,
        'objects': object_to_id
    }


def split_handler(splits):
    unique_splits = set()
    for split in splits:
        if split == 'testdev':
            unique_splits.add(split)
        else:
            for valid in valid_splits:
                if valid in split:
                    unique_splits.add(valid)
    return sorted(unique_splits)


def init_v_loader():
    pass

def init_vl_loader():
    pass

def init_loaders():
    pass


if __name__ == '__main__':
    config = Config()
    result = init_datasets(config)
