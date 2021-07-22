from collections import defaultdict


from vltk.features import Features
from vltk import adapters
import vltk.vars as vltk


class Coco2014(adapters.VisnDataset):
    @staticmethod
    def schema():
        return {
            vltk.box: Features.Box(),
            vltk.polygons: Features.Polygons(),
            vltk.objects: Features.StringList(),
        }

    @staticmethod
    def forward(json_files, splits):

        total_annos = {}
        id_to_cat = {}
        file_to_id_to_stem = defaultdict(dict)
        for file, json in json_files.items():
            if "instance" not in file:
                continue
            info = json["images"]
            for i in info:
                img_id = i["file_name"].split(".")[0]
                file_to_id_to_stem[file][i["id"]] = img_id
        for file, json in json_files.items():
            if "instance" not in file:
                continue

            categories = json["categories"]
            for cat in categories:
                id_to_cat[cat["id"]] = cat["name"]

            for entry in json["annotations"]:
                # TODO: change this image ID thing later

                img_id = str(file_to_id_to_stem[file][entry["image_id"]])
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
                    img_entry[vltk.objects].append(category_id)
                    img_entry[vltk.box].append(bbox)
                    img_entry[vltk.polygons].append(seg_mask)
                    total_annos[img_id] = img_entry
                else:
                    total_annos[img_id][vltk.box].append(bbox)
                    total_annos[img_id][vltk.objects].append(category_id)
                    total_annos[img_id][vltk.polygons].append(seg_mask)

        return [{vltk.imgid: img_id, **entry} for img_id, entry in total_annos.items()]
