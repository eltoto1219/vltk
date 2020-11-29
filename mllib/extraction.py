import os
# import numpy as np
from collections import OrderedDict

import datasets
import numpy as np
import torch
from tqdm import tqdm

from .legacy_utils import Config
from .modeling_frcnn import GeneralizedRCNN
from .processing_image import Preprocess


def schema_factory(max_detections):
    return datasets.Features(
        OrderedDict(
            {
                "attr_ids": datasets.Sequence(
                    length=max_detections, feature=datasets.Value("float32")
                ),
                "attr_probs": datasets.Sequence(
                    length=max_detections, feature=datasets.Value("float32")
                ),
                "boxes": datasets.Array2D((max_detections, 4), dtype="float32"),
                "img_id": datasets.Value("int32"),
                "obj_ids": datasets.Sequence(
                    length=max_detections, feature=datasets.Value("float32")
                ),
                "obj_probs": datasets.Sequence(
                    length=max_detections, feature=datasets.Value("float32")
                ),
                "roi_features": datasets.Array2D(
                    (max_detections, 2048), dtype="float32"
                ),
                "sizes": datasets.Sequence(length=2, feature=datasets.Value("float32")),
                "preds_per_image": datasets.Value(dtype="int32"),
            }
        )
    )


class Extract:
    def __init__(self, env, config, ids=None):

        self.env = env
        self.config = config
        if ids is not None:
            self.ids = set([x.replace("\n", "") for x in open(ids).readlines()])
        if self.env.gpus == -1:
            self.device = "cpu"
        else:
            self.device = "cuda"
        self.batch_size = self.config.batch_size
        self.data_dir = os.path.join(self.env.data_dir, self.config.input_dir)
        assert os.path.isdir(self.data_dir)
        self.output_file = os.path.join(self.env.data_dir, self.config.out_file)
        print(f"will write to: {self.output_file}")
        if self.batch_size == 0:
            self.batch_size = 1
        self.model_config = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.model = GeneralizedRCNN.from_pretrained(
            "unc-nlp/frcnn-vg-finetuned", config=self.model_config
        )
        self.model.to(self.device)
        self.preprocess = Preprocess(self.model.config)
        self.schema = schema_factory(self.model.config.max_detections)

    def get_files(self):
        file_list = []
        for path, subdirs, files in os.walk(self.data_dir):
            for name in files:
                file_list.append(os.path.join(path, name))
        return file_list

    @property
    def file_generator(self):
        batch = []
        files = self.get_files()
        self.progress = tqdm(unit_scale=True, desc="Extracting", total=len(files))
        for file in files:
            file = os.path.join(self.data_dir, file)
            file_id = file.split("/")[-1].split(".")[0]
            if hasattr(self, "ids") and file_id not in self.ids:
                continue
            batch.append((file_id, file))
            if len(batch) == self.batch_size:
                temp = batch
                batch = []
                yield list(map(list, zip(*temp)))

        for i in range(1):
            yield list(map(list, zip(*batch)))

    def __call__(self):
        # make writer
        if os.path.isfile(self.output_file):
            os.remove(self.output_file)
        writer = datasets.ArrowWriter(features=self.schema, path=self.output_file)
        # do file generator
        for img_ids, filepaths in self.file_generator:
            img_ids, images, sizes, scales_yx = self.preprocess(filepaths, img_ids)

            if torch.cuda.is_available():
                images, sizes, scales_yx = (
                    images.to(self.device),
                    sizes.to(self.device),
                    scales_yx.to(self.device),
                )
            output_dict = self.model(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=self.model.config.MAX_DETECTIONS,
                pad_value=0,
                return_tensors="np",
                location="cpu",
            )
            output_dict["boxes"] = output_dict.pop("normalized_boxes")
            output_dict["img_id"] = np.array(img_ids)
            batch = self.schema.encode_batch(output_dict)
            try:
                writer.write_batch(batch)
            except Exception:
                for k, v in output_dict.items():
                    print(k, len(v))
                raise Exception(f"all ids: {output_dict['img_id']}")
            self.progress.update(self.batch_size)

        num_examples, num_bytes = writer.finalize()
        print(f"Success! You wrote {num_examples} entry(s) and {num_bytes >> 20} mb")


# add the following to tests later
"""
if __name__ == "__main__":
    extract = Extract(sys.argv[1:])
    extract()
    if not TEST:
        dataset = datasets.Dataset.from_file(extract.outputfile)
        # wala!
        #print(np.array(dataset[0:2]["roi_features"]).shape)
"""
