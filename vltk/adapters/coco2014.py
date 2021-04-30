from collections import defaultdict

import vltk
from vltk import Features, adapters


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
                id_to_size[i["file_name"].split(".")[0]] = [
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
                # TODO: change this image ID thing later
                img_id = str(entry["image_id"])
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
