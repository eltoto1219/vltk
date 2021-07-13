from collections import defaultdict

import numpy as np
import vltk.vars as vltk
from vltk import adapters
from vltk.features import Features


class CLEVR(adapters.VisnDataset):
    @staticmethod
    def schema(dim=3):
        return {
            "positions": Features.Features2D(d=dim),
            "colors": Features.StringList(),
            "shapes": Features.StringList(),
            "sizes": Features.StringList(),
            "materials": Features.StringList(),
        }

    @staticmethod
    def forward(json_files, splits):
        entries = defaultdict(dict)
        for filepath, js in json_files.items():
            for scene in js["scenes"]:
                img_filename = scene["image_filename"]
                imgid = img_filename.split(".")[0]
                objects = scene["objects"]
                colors, shapes, materials, sizes, positions = [], [], [], [], []
                for obj in objects:
                    colors.append(obj["color"])
                    shapes.append(obj["shape"])
                    materials.append(obj["material"])
                    sizes.append(obj["size"])
                    positions.append(obj["pixel_coords"])
                entries[imgid] = {
                    "positions": np.array(positions),
                    "colors": colors,
                    "shapes": shapes,
                    "materials": materials,
                    "sizes": sizes,
                    vltk.imgid: imgid,
                }

        return [v for v in entries.values()]
