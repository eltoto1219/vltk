from collections import Counter

import vltk
from vltk import Features, adapters, compat
from vltk.adapters import Adapters
from vltk.configs import DataConfig, LangConfig, VisionConfig
from vltk.loader import build
from vltk.modeling.frcnn import FRCNN as FasterRCNN
from vltk.processing.label import label_default


# Visual Adatper
class FRCNN(adapters.VisnExtraction):

    default_processor = VisionConfig(
        **{
            "transforms": ["FromFile", "ToTensor", "ResizeTensor", "Normalize"],
            "size": (800, 1333),
            "mode": "bilinear",
            "pad_value": 0.0,
            "mean": [102.9801 / 255, 115.9465 / 255, 122.7717 / 255],
            "sdev": [1.0, 1.0, 1.0],
        }
    )
    weights = "unc-nlp/frcnn-vg-finetuned"
    model = FasterRCNN
    model_config = compat.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

    def schema(max_detections=36, visual_dim=2048):
        return {
            "attr_ids": Features.ids,
            "object_ids": Features.ids,
            vltk.features: Features.features(max_detections, visual_dim),
            vltk.boxtensor: Features.boxtensor(max_detections),
        }

    def forward(model, entry):

        size = entry["size"]
        scale_hw = entry["scale"]
        image = entry["image"]

        model_out = model(
            images=image.unsqueeze(0),
            image_shapes=size.unsqueeze(0),
            scales_yx=scale_hw.unsqueeze(0),
            padding="max_detections",
            pad_value=0.0,
            return_tensors="np",
            location="cpu",
        )
        return {
            "object_ids": model_out["obj_ids"],
            "attr_ids": model_out["attr_ids"],
            vltk.boxtensor: model_out["normalized_boxes"],
            vltk.features: model_out["roi_features"],
        }


# Vision Datasets


class VisualGenome(adapters.VisnDataset):
    def schema():
        return {}

    def forward(json_files, splits):
        return {}


class CLEVR(adapters.VisnDataset):
    def schema():
        pass

    def forward(json_files, splits):
        pass


# Vision-Language Datasets


class GQA(adapters.VisnLangDataset):
    data_info = {
        "dev": {"coco2014": ["test"]},
        "train": {"visualgenome": ["train"]},
        "val": {"visualgenome": ["train"]},
        "test": {"coco2014": ["test"]},
        "testdev": {"coco2014": ["val"]},
    }

    def schema():
        return {}

    def forward(json_files, split, min_label_frequency=2):
        skipped = 0
        label_frequencies = Counter()
        batch_entries = []

        for t in json_files:
            for i, (k, v) in enumerate(t.items()):
                if "answer" in v:
                    answer = label_default(v["answer"])
                    label_frequencies.update([answer])

            for i, (k, v) in enumerate(t.items()):
                if split == "test":
                    answer = None
                elif label_frequencies[v["answer"]] < min_label_frequency:
                    skipped += 1
                    continue
                else:
                    answer = label_default(v["answer"])

                text = v["question"]
                img_id = v["imageId"].lstrip("n")
                entry = {
                    vltk.text: text,
                    vltk.imgid: img_id,
                    vltk.label: [answer],
                    vltk.score: [1.0],
                }

                batch_entries.append(entry)

        return batch_entries


if __name__ == "__main__":
    # set datadir
    datadir = "/home/eltoto/demodata"
    # create datasets
    # cocofeats = FRCNN.extract(datadir, dataset_name="coco2014")
    # feats = FRCNN.load("/home/eltoto/demodata/coco2014/frcnn/val.arrow")
    # feats = FRCNN.load("/home/eltoto/demodata/", dataset_name="coco2014", split="val")
    # vgfeats = FRCNN.extract(datadir, dataset_name="visualgenome")
    # coco2014 = Adapters().get("coco2014").extract(datadir)
    # annos = coco2014 = Coco2014.load(datadir)
    # print(annos)
    # visualgenome = VisualGenome.extract(datadir)
    # vqa = Adapters().get("vqa").extract(datadir)
    # gqa = GQA.extract(datadir)
    # gqa = GQA.load(datadir, split="train")
    # print(gqa)
    # add adapters
    Adapters().add(GQA, VisualGenome, FRCNN)
    # print(Adapters().avail())
    # superset datasets
    # define config for dataset
    config = DataConfig(
        lang=LangConfig(tokenizer="BertWordPieceTokenizer"),
        # choose which dataset and dataset split for train and eval
        train_datasets=[["vqa", "trainval"], ["gqa", "train"]],
        # eval_datasets=["gqa", "testdev"],
        # choose which tokenizer to use
        # choose which feature extractor to use
        extractor=None,
        datadir=datadir,
        train_batch_size=2,
        eval_batch_size=2,
        img_first=True,
    )

    train_loader, val_loader = build(config)
    for x in train_loader:
        print(x)
        break
    # first entry in the dataset
