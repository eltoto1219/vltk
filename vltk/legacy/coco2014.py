import timeit
from collections import defaultdict

import datasets as ds
import vltk.vars as vltk
from tqdm import tqdm
from vltk.adapters import VisnDataset
from vltk.processing.label import clean_imgid_default

# ignore 'iscrowd' because it used rle


class Coco2014(VisnDataset):
    @staticmethod
    def default_features(**kwargs):
        return {
            "bbox": ds.Sequence(
                length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
            ),
            "segmentation": ds.Sequence(
                length=-1,
                feature=ds.Sequence(
                    length=-1,
                    feature=ds.Sequence(length=-1, feature=ds.Value("float32")),
                ),
            ),
            "area": ds.Sequence(length=-1, feature=ds.Value("float32")),
            "size": ds.Sequence(length=-1, feature=ds.Value("int16")),
        }

    # add code to ensure correct format of all bounding boxes
    def forward(json_files):
        """ json_files: list(tuple(str, dict)) """

        total_annos = {}
        id_to_cat = {}
        id_to_size = {}
        for file, json in tqdm(json_files):
            if "instance" not in file:
                continue
            info = json["images"]
            for i in info:
                id_to_size[clean_imgid_default(i["file_name"]).split(".")[0]] = [
                    i["height"],
                    i["width"],
                ]

        for file, json in tqdm(json_files):
            if "instance" not in file:
                continue

            categories = json["categories"]
            for cat in categories:
                id_to_cat[cat["id"]] = cat["name"]

            for entry in tqdm(json["annotations"]):
                img_id = clean_imgid_default(str(entry["image_id"]))
                entry["size"] = id_to_size[img_id]
                bbox = entry["bbox"]
                area = entry["area"]
                segmentation = entry["segmentation"]
                category_id = id_to_cat[entry["category_id"]]
                if entry["iscrowd"]:
                    seg_mask = []
                else:
                    seg_mask = segmentation
                    if not isinstance(seg_mask[0], list):
                        seg_mask = [seg_mask]
                # seg_mask, size = compress_mask(
                #     seg_to_mask(segmentation, entry["size"][0], entry["size"][1])
                # )

                img_data = total_annos.get(img_id, None)
                if img_data is None:
                    img_entry = defaultdict(list)
                    img_entry["size"] = entry["size"]
                    img_entry[vltk.label].append(category_id)
                    img_entry["bbox"].append(bbox)
                    img_entry["area"].append(area)
                    img_entry["segmentation"].append(seg_mask)
                    total_annos[img_id] = img_entry
                else:
                    total_annos[img_id]["bbox"].append(bbox)
                    total_annos[img_id][vltk.label].append(category_id)
                    total_annos[img_id]["area"].append(area)
                    total_annos[img_id]["segmentation"].append(seg_mask)
                    # for sm in seg_mask:

        return [{vltk.imgid: img_id, **entry} for img_id, entry in total_annos.items()]


if __name__ == "__main__":

    coco = Coco2014.extract(
        searchdirs="/playpen1/home/avmendoz/data",
    )

    start = timeit.timeit()
    coco = Coco2014.load(
        "/playpen1/home/avmendoz/data/coco2014/annotations/annotations.arrow"
    )
    end = timeit.timeit()
    print(coco, f"duration to load: {end - start} s")

    print("example:", coco[0]["label"])
