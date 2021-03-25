import os
from collections import Counter, defaultdict

import vltk
from vltk import Features, compat
from vltk.abc.extraction import VizExtractionAdapter, VizExtractionAdapters
from vltk.abc.visnadapter import VisnDatasetAdapter, VisnDatasetAdapters
from vltk.abc.visnlangadatper import (VisnLangDatasetAdapter,
                                      VisnLangDatasetAdapters)
from vltk.configs import DataConfig, ProcessorConfig
from vltk.loader.builder import init_datasets
from vltk.metrics import soft_score
from vltk.modeling.frcnn import FRCNN as FasterRCNN
from vltk.processing.label import clean_imgid_default


# Visual Adatper
class FRCNN(VizExtractionAdapter):

    default_processor = ProcessorConfig(
        **{
            "transforms": ["ToPILImage", "ToTensor", "ResizeTensor", "Normalize"],
            "size": (800, 1333),
            "mode": "bilinear",
            "pad_value": 0.0,
            "mean": [102.9801, 115.9465, 122.7717],
            "sdev": [1.0, 1.0, 1.0],
        }
    )
    model_config = compat.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    weights = "unc-nlp/frcnn-vg-finetuned"
    model = FasterRCNN

    def schema(max_detections=36, visual_dim=2048):
        return {
            "attr_ids": Features.ids,
            "object_ids": Features.ids,
            vltk.features: Features.features(max_detections, visual_dim),
            vltk.boxtensor: Features.boxtensor(max_detections),
        }

    def forward(model, entry, **kwargs):

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
class Coco2014(VisnDatasetAdapter):
    def schema():
        return {vltk.box: Features.box, vltk.segmentation: Features.segmentation}

    def forward(json_files, **kwargs):

        total_annos = {}
        id_to_cat = {}
        id_to_size = {}
        for file, json in json_files:
            if "instance" not in file:
                continue
            info = json["images"]
            for i in info:
                id_to_size[clean_imgid_default(i["file_name"]).split(".")[0]] = [
                    i["height"],
                    i["width"],
                ]
        for file, json in json_files:
            if "instance" not in file:
                continue

            categories = json["categories"]
            for cat in categories:
                id_to_cat[cat["id"]] = cat["name"]

            for entry in json["annotations"]:
                img_id = clean_imgid_default(str(entry["image_id"]))
                bbox = entry["bbox"]
                segmentation = entry["segmentation"]
                category_id = id_to_cat[entry["category_id"]]
                if entry["iscrowd"]:
                    seg_mask = []
                else:
                    seg_mask = segmentation
                    if not isinstance(seg_mask[0], list):
                        seg_mask = [seg_mask]
                img_data = total_annos.get(img_id, None)
                if img_data is None:
                    img_entry = defaultdict(list)
                    img_entry[vltk.label].append(category_id)
                    img_entry[vltk.box].append(bbox)
                    img_entry[vltk.segmentation].append(seg_mask)
                    total_annos[img_id] = img_entry
                else:
                    total_annos[img_id][vltk.box].append(bbox)
                    total_annos[img_id][vltk.label].append(category_id)
                    total_annos[img_id][vltk.segmentation].append(seg_mask)

        return [{vltk.imgid: img_id, **entry} for img_id, entry in total_annos.items()]


class VisualGenome(VisnDatasetAdapter):
    def schema():
        return {}

    def forward(json_files, **kwargs):
        return {}


# Vision-Language Datasets
class VQA(VisnLangDatasetAdapter):
    data_info = {
        "val": {"coco2014": ["val"]},
        "train": {"coco2014": ["train"]},
        "test": {"coco2014": ["test"]},
    }
    schema = {"qid": Features.string}

    def forward(json_files, split, **kwargs):
        min_label_frequency = kwargs.get("min_label_frequency")
        batch_entries = []
        all_questions = []
        qid2answers = {}
        label_frequencies = Counter()
        label_preprocessor = kwargs.get("label_preprocessor", None)
        if label_preprocessor is None:

            def label_preprocessor(x):
                return x

        for x in json_files:
            if "questions" in x:
                all_questions.extend(x["questions"])
            else:
                annotations = x["annotations"]
                accepted_answers = {
                    label_preprocessor(anno["multiple_choice_answer"])
                    for anno in annotations
                }
                for anno in annotations:
                    qid = str(anno["question_id"])
                    answers = anno["answers"]
                    label_frequencies.update(
                        [label_preprocessor(anno["multiple_choice_answer"])]
                    )
                    answer_counter = Counter()
                    for ans_dict in answers:
                        ans = ans_dict["answer"]
                        if ans not in accepted_answers:
                            pass
                        else:
                            ans = label_preprocessor(ans)
                            answer_counter.update([ans])
                    qid2answers[qid] = {
                        k: soft_score(v) for k, v in answer_counter.items()
                    }

        skipped = 0
        for entry in all_questions:
            entry[vltk.imgid] = str(entry.pop("image_id"))
            entry[vltk.text] = entry.pop("question")
            entry["qid"] = str(entry.pop("question_id"))
            try:
                entry[VisnLangDatasetAdapter.label_key] = qid2answers[entry["qid"]]
                labels = {
                    l: s
                    for l, s in entry[VisnLangDatasetAdapter.label_key].items()
                    if label_frequencies[l] > min_label_frequency
                }
                if not labels:
                    skipped += 1
                    continue

                labels, scores = VisnLangDatasetAdapter._label_handler(labels)
                entry[vltk.score] = scores
                entry[vltk.label] = labels
            except KeyError:
                pass

            batch_entries.append(entry)
        print(f"SKIPPEd {skipped} entries")
        return batch_entries


class GQA(VisnLangDatasetAdapter):
    data_info = {
        "dev": {"coco2014": ["test"]},
        "train": {"visualgenome": ["train"]},
        "val": {"visualgenome": ["train"]},
        "test": {"coco2014": ["test"]},
        "testdev": {"coco2014": ["val"]},
    }
    schema = {}

    def forward(json_files, split, **kwargs):
        skipped = 0
        min_label_frequency = kwargs.get("min_label_frequency", 2)
        label_preprocessor = kwargs.get("label_preprocessor", None)
        label_frequencies = Counter()
        batch_entries = []
        if label_preprocessor is None:

            def label_preprocessor(x):
                return x

        for t in json_files:
            for i, (k, v) in enumerate(t.items()):
                if "answer" in v:
                    answer = label_preprocessor(v["answer"])
                    label_frequencies.update([answer])

            for i, (k, v) in enumerate(t.items()):
                if split == "test":
                    answer = None
                elif label_frequencies[v["answer"]] < min_label_frequency:
                    skipped += 1
                    continue
                else:
                    answer = label_preprocessor(v["answer"])

                text = v["question"]
                img_id = v["imageId"].lstrip("n")
                entry = {
                    vltk.text: text,
                    vltk.imgid: img_id,
                    vltk.label: [answer],
                    vltk.score: [1.0],
                }

                batch_entries.append(entry)

        print(f"SKIPPEd {skipped} entries")
        return batch_entries


if __name__ == "__main__":
    # set datadir
    datadir = "/home/eltoto/demodata"
    # create datasets
    # cocofeats = FRCNN.extract(datadir, dataset_name="coco2014")
    # vgfeats = FRCNN.extract(datadir, dataset_name="visualgenome")
    # coco2014 = Coco2014.extract(datadir)
    # visualgenome = VisualGenome.extract(datadir)
    # vqa = VQA.extract(datadir)
    # gqa = GQA.extract(datadir)
    # add adapters
    Vizlang = VisnLangDatasetAdapters()
    Viz = VisnDatasetAdapters()
    Extract = VizExtractionAdapters()
    Vizlang.add(VQA)
    Vizlang.add(GQA)
    Viz.add(Coco2014)
    Viz.add(VisualGenome)
    Extract.add(FRCNN)

    # superset datasets
    # define config for dataset
    config = DataConfig(
        # choose which dataset and dataset split for train and eval
        train_datasets=[["gqa", "train"], ["vqa", "trainval"]],
        eval_datasets=["gqa", "testdev"],
        # choose which tokenizer to use
        tokenizer="BertWordPeice",
        # choose which feature extractor to use
        extractor="frcnn",
        datadir=datadir,
        train_batch_size=1,
        eval_batch_size=1,
        img_first=True,
    )
    # use config to create dataset
    (train, val), _, answer_to_id, object_to_id = init_datasets(config)
    train_loader = train[1]
    for x in train_loader:
        print(x.keys())
        break
    # first entry in the dataset
