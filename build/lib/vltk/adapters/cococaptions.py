import vltk.vars as vltk
from tqdm import tqdm
from vltk import adapters


class COCOCaptions(adapters.VisnLangDataset):
    data_info = {
        "train": {"coco2014": ["train"]},
        "val": {"coco2014": ["val"]},
    }

    @staticmethod
    def schema():
        return {}

    @staticmethod
    def forward(json_files, split, min_label_frequency=2):
        batch_entries = []
        id2imgid = {}
        for filename, data in json_files.items():
            if "annotations" not in data:
                continue
            if "caption" not in data["annotations"][0]:
                continue
            for img in data["images"]:
                id2imgid[img["id"]] = img["file_name"]
            for item in tqdm(data["annotations"]):
                imgid = id2imgid[item["image_id"]].split(".")[0]
                entry = {vltk.imgid: imgid, vltk.text: item["caption"]}

                batch_entries.append(entry)

        return batch_entries
