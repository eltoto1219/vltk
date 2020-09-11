import os
import tempfile
from collections import OrderedDict

import datasets
import numpy as np

from frcnn import Config, GeneralizedRCNN, Preprocess


# import numpy as np


TEST = True
CONFIG = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
BATCH = 32
URL = "https://vqa.cloudcv.org/media/test2014/COCO_test2014_000000168437.jpg"
DEFAULT_SCHEMA = datasets.Features(
    OrderedDict(
        {
            "attr_ids": datasets.Sequence(
                length=CONFIG.MAX_DETECTIONS, feature=datasets.Value("float32")
            ),
            "attr_probs": datasets.Sequence(
                length=CONFIG.MAX_DETECTIONS, feature=datasets.Value("float32")
            ),
            "boxes": datasets.Array2D((CONFIG.MAX_DETECTIONS, 4), dtype="float32"),
            "iid": datasets.Value("int32"),
            "obj_ids": datasets.Sequence(
                length=CONFIG.MAX_DETECTIONS, feature=datasets.Value("float32")
            ),
            "obj_probs": datasets.Sequence(
                length=CONFIG.MAX_DETECTIONS, feature=datasets.Value("float32")
            ),
            "roi_features": datasets.Array2D(
                (CONFIG.MAX_DETECTIONS, 2048), dtype="float32"
            ),
            "sizes": datasets.Sequence(length=2, feature=datasets.Value("float32")),
            "preds_per_image": datasets.Value(dtype="int32"),
        }
    )
)


def get_images(path=URL):
    if not os.path.isdir(path):
        if not isinstance(path, list):
            listdir = [path]
    else:
        listdir = os.listdir(os.getcwd())
    for i in range(len(listdir)):
        string = listdir.pop(i)
        cur = os.path.join(path, string)
        if not os.path.isfile(cur):
            cur = string
        listdir.insert(i, (i, cur))
    return listdir


def chunk(images, batch=BATCH):
    return (images[i : i + batch] for i in range(0, len(images), batch))


if __name__ == "__main__":
    # init model
    model = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    image_processor = Preprocess(CONFIG)

    # open writer
    with tempfile.TemporaryDirectory() as tmp_dir:
        writer = datasets.ArrowWriter(
            features=DEFAULT_SCHEMA, path=os.path.join(tmp_dir, "temp.arrow")
        )

        # extract the data
        for imgs in chunk(get_images()):
            ids, batch = list(map(list, zip(*imgs)))
            images, sizes, scales_yx = image_processor(batch)
            output_dict = model(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=CONFIG.MAX_DETECTIONS,
                pad_value=0,
                return_tensors="np",
            )
            if TEST:
                output_dict["iid"] = [np.random.randint(1)] * len(
                    output_dict["preds_per_image"]
                )

            batch = DEFAULT_SCHEMA.encode_batch(output_dict)
            writer.write_batch(batch)

        # finalizer the writer
        num_examples, num_bytes = writer.finalize()
        dataset = datasets.Dataset.from_file(os.path.join(tmp_dir, "temp.arrow"))

# and wala!!!
print(np.array(dataset[0:2]["roi_features"]).shape)
