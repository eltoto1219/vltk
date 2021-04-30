from collections import defaultdict

from tqdm import tqdm

import vltk
from vltk import Features, adapters
from vltk.adapters import Adapters
from vltk.configs import DataConfig
from vltk.loader.builder import init_datasets


# data source: https://github.com/ccvl/clevr-refplus-dataset-gen
class CLEVRREF(adapters.VisnDataset):
    def schema():
        return {
            vltk.segmentation: Features.Segmentation,
            "colors": Features.StringList,
            "shapes": Features.StringList,
            "sizes": Features.StringList,
            "materials": Features.StringList,
            vltk.box: Features.Box,
        }

    def forward(json_files, splits):
        # default box order: x, y, h, w
        entries = defaultdict(dict)
        for filepath, js in json_files:
            if "scene" not in filepath:
                continue
            for scene in tqdm(js["scenes"]):
                img_filename = scene["image_filename"]
                imgid = img_filename.split(".")[0]
                colors = []
                shapes = []
                materials = []
                sizes = []
                boxes = []
                segmentations = []
                for idx, (obj, bbox, seg) in enumerate(
                    zip(
                        scene["objects"],
                        scene["obj_bbox"].values(),
                        scene["obj_mask"].values(),
                    )
                ):
                    boxes.append(bbox)
                    colors.append(obj["color"])
                    shapes.append(obj["shape"])
                    materials.append(obj["material"])
                    sizes.append(obj["size"])
                    try:
                        seg = list(eval(seg))
                    except Exception:
                        continue

                    segmentations.append([seg])

                entries[imgid] = {
                    vltk.segmentation: segmentations,
                    "colors": colors,
                    "shapes": shapes,
                    "materials": materials,
                    "sizes": sizes,
                    vltk.box: boxes,
                    vltk.imgid: imgid,
                }

        return [v for v in entries.values()]


if __name__ == "__main__":
    # set datadir
    datadir = "/home/eltoto/demodata"
    # create datasets
    # clevrref = CLEVRREF.extract(datadir, ignore_files="exp")
    # cocofeats = FRCNN.extract(datadir, dataset_name="coco2014")
    # feats = FRCNN.load("/home/eltoto/demodata/coco2014/frcnn/val.arrow")
    # feats = FRCNN.load("/home/eltoto/demodata/", dataset_name="coco2014", split="val")
    # vgfeats = FRCNN.extract(datadir, dataset_name="visualgenome")
    # coco2014 = Coco2014.extract(datadir)
    # annos = coco2014 = Coco2014.load(datadir)
    # print(annos)
    # visualgenome = VisualGenome.extract(datadir)
    # vqa = VQA.extract(datadir)
    # gqa = GQA.extract(datadir)
    # gqa = GQA.load(datadir, split="train")
    # print(gqa)
    # add adapters
    Adapters().add(CLEVRREF)
    # print(Adapters().avail())
    # superset datasets
    # define config for dataset
    config = DataConfig(
        # choose which dataset and dataset split for train and eval
        train_datasets=[
            ["clevrref", "trainval"],
        ],
        # eval_datasets=["gqa", "testdev"],
        # choose which tokenizer to use
        tokenizer="BertWordPieceTokenizer",
        # choose which feature extractor to use
        extractor=None,
        datadir=datadir,
        train_batch_size=1,
        eval_batch_size=1,
        img_first=True,
    )
    # # use config to create dataset
    (train, val), _, answer_to_id, object_to_id = init_datasets(config)
    train_loader = train[1]
    for x in train_loader:
        print(x)
        break
    # first entry in the dataset
