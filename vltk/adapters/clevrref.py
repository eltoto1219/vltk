from collections import defaultdict

import vltk.vars as vltk
from tqdm import tqdm
from vltk import adapters
from vltk.adapters import Adapters
from vltk.configs import DataConfig
from vltk.features import Features


# data source: https://github.com/ccvl/clevr-refplus-dataset-gen
# https://cs.jhu.edu/~cxliu/data/clevr_ref+_1.0.zip
class CLEVRREF(adapters.VisnDataset):
    @staticmethod
    def schema():
        return {
            vltk.RLE: Features.RLE(),
            "colors": Features.StringList(),
            "shapes": Features.StringList(),
            "sizes": Features.StringList(),
            "materials": Features.StringList(),
            vltk.box: Features.Box(),
        }

    @staticmethod
    def forward(json_files, splits):
        # default box order: x, y, h, w
        entries = defaultdict(dict)
        for filepath, js in json_files.items():
            if "scene" not in filepath:
                continue
            for i, scene in enumerate(tqdm(js["scenes"])):
                img_filename = scene["image_filename"]
                imgid = img_filename.split(".")[0]
                colors = []
                shapes = []
                materials = []
                sizes = []
                boxes = []
                points = []
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

                        seg = seg.split(",")
                    except Exception:
                        continue

                    points.append(seg)

                entries[imgid] = {
                    vltk.RLE: points,
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
    print(Adapters().avail())

    # TODO: make sure visnlang adapters are being added
    Adapters().add(CLEVRREF)
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
