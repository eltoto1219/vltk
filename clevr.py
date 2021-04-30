from collections import defaultdict

import numpy as np

import vltk
from vltk import Features, adapters
from vltk.adapters import Adapters
from vltk.configs import DataConfig
from vltk.loader.builder import init_datasets


class CLEVR(adapters.VisnDataset):
    def schema(dim=3):
        return {
            "positions": Features.Features2D(d=dim),
            "colors": Features.StringList,
            "shapes": Features.StringList,
            "sizes": Features.StringList,
            "materials": Features.StringList,
        }

    def forward(json_files, splits):
        entries = defaultdict(dict)
        for filepath, js in json_files:
            for scene in js["scenes"]:
                img_filename = scene["image_filename"]
                imgid = img_filename.split(".")[0]
                objects = scene["objects"]
                colors = []
                shapes = []
                materials = []
                sizes = []
                segmentations = []
                for obj in objects:
                    colors.append(obj["color"])
                    shapes.append(obj["shape"])
                    materials.append(obj["material"])
                    sizes.append(obj["size"])
                    segmentations.append(obj["pixel_coords"])

                entries[imgid] = {
                    "positions": np.array(segmentations),
                    "colors": colors,
                    "shapes": shapes,
                    "materials": materials,
                    "sizes": sizes,
                    vltk.imgid: imgid,
                }

        return [v for v in entries.values()]


if __name__ == "__main__":
    # set datadir
    datadir = "/home/eltoto/demodata"
    # create datasets
    clevr = CLEVR.extract(datadir)
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
    Adapters().add(CLEVR)
    # print(Adapters().avail())
    # superset datasets
    # define config for dataset
    config = DataConfig(
        # choose which dataset and dataset split for train and eval
        train_datasets=[
            ["clevr", "trainval"],
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
