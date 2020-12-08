import os
import pickle
# import numpy as np
from collections import OrderedDict
from copy import deepcopy

import datasets
import numpy as np
import torch
from datasets import ArrowWriter, Features, Split
from pyarrow import OSFile
from tqdm import tqdm

from .legacy_processing import Preprocess


def raw_feat_factory(max_size, min_size):
    return Features(
        OrderedDict(
            {
                "img_features": datasets.Array4D(
                    tuple([1, 3, 832, 832]), dtype="uint8"
                ),
                "img_id": datasets.Value("string"),
                "sizes": datasets.Sequence(length=2, feature=datasets.Value("int16")),
                "raw_sizes": datasets.Sequence(
                    length=2, feature=datasets.Value("int16")
                ),
            }
        )
    )


def feature_factory(max_detections):
    return Features(
        OrderedDict(
            {
                "attr_ids": datasets.Sequence(
                    length=max_detections, feature=datasets.Value("float32")
                ),
                "attr_probs": datasets.Sequence(
                    length=max_detections, feature=datasets.Value("float32")
                ),
                "boxes": datasets.Array2D((max_detections, 4), dtype="float32"),
                "img_id": datasets.Value("string"),
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
    def __init__(self, model, preproc, env, config, ids=None):

        self.env = env
        self.config = config
        if ids is not None:
            self.ids = set([x.replace("\n", "") for x in open(ids).readlines()])
        if self.env.gpus == -1:
            self.device = "cpu"
        else:
            self.device = f"cuda:{self.env.gpus}"
        self.data_dir = os.path.join(self.env.data_dir, self.config.input_dir)
        assert os.path.isdir(self.data_dir)
        self.output_file = os.path.join(self.env.data_dir, self.config.out_file)
        print(f"will write to: {self.output_file}")
        self.model = model
        if model is not None:
            self.model.to(self.device)
        if preproc is not None:
            self.preprocess = preproc
            self.features = None
        else:
            self.preprocess = Preprocess(self.model.config)
        if model is not None:
            self.features = feature_factory(self.model.config.max_detections)
        else:
            pass
        self.skipped_ids = set()
        files, dirs = self.get_files()
        self.files = files
        self.subdirs = dirs
        if len(self.subdirs) == 0:
            self.subdirs = [self.config.input_dir.split("/")[-1]]
        self.progress = tqdm(unit_scale=True, desc="Extracting", total=len(self.files))

    def get_files(self):
        file_list = []
        dirs = []
        for path, subdirs, files in os.walk(self.data_dir):
            for sdirs in subdirs:
                dirs.append(sdirs)
            for name in files:
                file = os.path.join(path, name)
                file_id = file.split("/")[-1].split(".")[0]
                if hasattr(self, "ids") and file_id not in self.ids:
                    continue
                file_list.append(file)
        return list(set(file_list)), list(set(dirs))

    def file_generator(self, split):
        for file in self.files:
            if split not in file:
                continue
            file = os.path.join(self.data_dir, file)
            file_id = file.split("/")[-1].split(".")[0]
            yield (file_id, file)

    def make_temp_name(self, split):
        if len(self.subdirs) > 1:
            temp = (
                f'{self.config.out_file.split(".")[0]}_{split.strip("/")}.temp'.replace(
                    "/", "_"
                )
            )
        else:
            temp = f'{self.config.out_file.split(".")[0]}.temp'.replace("/", "_")
        return temp

    def extract_torch_format(self, split, tempfile_or_buf):
        for z, (img_id, filepath) in enumerate(self.file_generator(split)):
            sdir = (
                self.output_file
                if split in self.output_file
                else os.path.join(self.output_file, split)
            )
            os.makedirs(sdir, exist_ok=True)
            fp = os.path.join(sdir, img_id + ".pt")
            if not os.path.isfile(fp):
                image, (ogh, ogw), (nh, nw) = self.preprocess(
                    filepath, min_size=832, max_size=832, use_gpu=True
                )
                if image is None:
                    continue
                image = image.type("torch.ByteTensor")
                assert image.shape == (3, 832, 832), image.shape
                assert not os.path.isfile(self.output_file), self.output_file

                torch.save(image, fp)
                self.progress.update(1)

    def extract_arrow_format(
        self, split, tempfile_or_buf, num_examples, num_bytes, dsets
    ):
        with OSFile(tempfile_or_buf, "wb") as s:
            writer = ArrowWriter(features=self.features, stream=s)
            # do file generator
            for z, (img_id, filepath) in enumerate(self.file_generator(split)):
                assert split in filepath, f"{(split, filepath)}"

                out_ids, images, sizes, scales_yx = self.preprocess(filepath, img_id)
                if not out_ids:
                    self.skipped_ids.add(img_id)
                    continue

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
                output_dict["img_id"] = [img_id]
                batch = self.features.encode_batch(output_dict)
                writer.write_batch(batch)
                self.progress.update(1)
            num_ex, num_b = writer.finalize()
            num_examples += num_ex
            num_bytes += num_b

        with OSFile(tempfile_or_buf, "rb") as s:
            dsets[split] = datasets.Dataset.from_buffer(s.read(), split=Split(split))

    def __call__(self):
        # make streams
        dsets = {}
        num_examples = 0
        num_bytes = 0
        temps = []
        for split in self.subdirs:
            temp = self.make_temp_name(split)
            temps.append(temp)
            if self.model is not None:
                self.extract_arrow_format(split, temp, num_examples, num_bytes, dsets)
            else:
                self.extract_torch_format(split, temp)

        # okay now we can get to the good part of combining datsetsk
        if self.model is not None:
            if len(dsets.values()) == 1:
                final = next(iter(dsets.values()))
                print(final)
            else:
                final = datasets.concatenate_datasets(
                    dsets=list(dsets.values()), split=[Split(x) for x in dsets]
                )
            final = pickle.loads(pickle.dumps(final))
            for y in temps:
                os.remove(y)
            writer = ArrowWriter(path=self.output_file)
            writer.write_table(final._data)
            e, b = writer.finalize()
            print(
                f"Success! You wrote {num_examples} entry(s) and {num_bytes >> 20} mb"
            )
            print(f"saved {self.output_file}")
            print(f"num ids skipped: {len(self.skipped_ids)}")
        else:
            print("done!")


"""
if __name__ == "__main__":
    extract = Extract(sys.argv[1:])
    extract()
    if not TEST:
        # wala!
        #print(np.array(dataset[0:2]["roi_features"]).shape)
"""
