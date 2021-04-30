from collections import Counter, defaultdict

import vltk
from vltk import Features, adapters, compat
from vltk.adapters import Adapters
from vltk.configs import DataConfig, ProcessorConfig
from vltk.loader.builder import init_datasets
from vltk.metrics import soft_score
from vltk.modeling.frcnn import FRCNN as FasterRCNN
from vltk.processing.label import clean_imgid_default, label_default


# Visual Adatper
class FRCNN(adapters.VisnExtraction):

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
    # model_config = compat.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
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
class Coco2014(adapters.VisnDataset):
    def schema():
        return {vltk.box: Features.box, vltk.segmentation: Features.segmentation}

    def forward(json_files, splits):

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
class VQA(adapters.VisnLangDataset):
    data_info = {
        "val": {"coco2014": ["val"]},
        "train": {"coco2014": ["train"]},
        "test": {"coco2014": ["test"]},
    }

    def schema():
        return {"qid": Features.string}

    def forward(json_files, split, min_label_frequency=9):
        batch_entries = []
        all_questions = []
        qid2answers = {}
        label_frequencies = Counter()
        for x in json_files:
            if "questions" in x:
                all_questions.extend(x["questions"])
            else:
                annotations = x["annotations"]
                accepted_answers = {
                    label_default(anno["multiple_choice_answer"])
                    for anno in annotations
                }
                for anno in annotations:
                    qid = str(anno["question_id"])
                    answers = anno["answers"]
                    label_frequencies.update(
                        [label_default(anno["multiple_choice_answer"])]
                    )
                    answer_counter = Counter()
                    for ans_dict in answers:
                        ans = ans_dict["answer"]
                        if ans not in accepted_answers:
                            pass
                        else:
                            ans = label_default(ans)
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
                entry[vltk.label] = qid2answers[entry["qid"]]
                labels = {
                    l: s
                    for l, s in entry[vltk.label].items()
                    if label_frequencies[l] > min_label_frequency
                }
                if not labels:
                    skipped += 1
                    continue

                labels, scores = adapters.VisnLangDataset._label_handler(labels)
                entry[vltk.score] = scores
                entry[vltk.label] = labels
            except KeyError:
                pass

            batch_entries.append(entry)
        return batch_entries


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
    # coco2014 = Coco2014.extract(datadir)
    # annos = coco2014 = Coco2014.load(datadir)
    # print(annos)
    # visualgenome = VisualGenome.extract(datadir)
    # vqa = VQA.extract(datadir)
    # gqa = GQA.extract(datadir)
    # gqa = GQA.load(datadir, split="train")
    # print(gqa)
    # add adapters
    raise Exception(Adapters().avail())
    Adapters().add(VQA, GQA, Coco2014, VisualGenome, FRCNN)
    # print(Adapters().avail())
    # superset datasets
    # define config for dataset
    config = DataConfig(
        # choose which dataset and dataset split for train and eval
        train_datasets=[
            ["coco2014", "trainval"],
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
