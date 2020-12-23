import os
import pickle
from collections import OrderedDict, defaultdict
from glob import glob
from io import BytesIO

import datasets
import torch
from datasets import ArrowWriter, Features, Split
from pyarrow import OSFile
from tqdm import tqdm

from mllib import compat
from mllib.models import frcnn
from mllib.proccessing import Image

MAX_DETECTIONS = 36
FEATUREDIM = 2048
BOXDIM = 4
MAX_IMG_SIZE = 832


def raw_feat_factory(max_img_size, channels):
    return Features(
        OrderedDict(
            {
                "img_features": datasets.Array3D(
                    tuple([channels, max_img_size, max_img_size]), dtype="uint8"
                ),
                "img_id": datasets.Value("string"),
                "sizes": datasets.Sequence(length=2, feature=datasets.Value("int16")),
                "raw_sizes": datasets.Sequence(
                    length=2, feature=datasets.Value("int16")
                ),
            }
        )
    )


def feat_factory(max_detections):
    return Features(
        OrderedDict(
            {
                "attr_ids": datasets.Sequence(
                    length=MAX_DETECTIONS, feature=datasets.Value("float32")
                ),
                "attr_probs": datasets.Sequence(
                    length=MAX_DETECTIONS, feature=datasets.Value("float32")
                ),
                "boxes": datasets.Array2D((MAX_DETECTIONS, BOXDIM), dtype="float32"),
                "img_id": datasets.Value("string"),
                "obj_ids": datasets.Sequence(
                    length=MAX_DETECTIONS, feature=datasets.Value("float32")
                ),
                "obj_probs": datasets.Sequence(
                    length=MAX_DETECTIONS, feature=datasets.Value("float32")
                ),
                "roi_features": datasets.Array2D(
                    (MAX_DETECTIONS, FEATUREDIM), dtype="float32"
                ),
                "sizes": datasets.Sequence(length=2, feature=datasets.Value("float32")),
                "preds_per_image": datasets.Value(dtype="int32"),
            }
        )
    )
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
    def __init__(self, config, ids=None):

        if ids is not None:
            self.ids = set([x.replace("\n", "") for x in open(ids).readlines()])
        if config.device == -1:
            self.device = "cpu"
        else:
            self.device = f"cuda:{config.device}"
        self.data_dir = config.pathes.data_dir
        self.output_file = config.extract.output_file
        print(f"will write to: {self.output_file}")
        self.model_config = compat.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.model = frcnn.FRCNN.from_pretrained(
            "unc-nlp/frcnn-vg-finetuned", config=self.model_config
        )
        self.model.to(self.device)
        self.features = feat_factory(self.config.max_detections)
        self.skipped_ids = set()
        self.subdir2fileidAndfile()

    def subdir2fileidAndfile(self):
        self.file_list = defaultdict(list)
        tempdirs = []
        streams = []
        file_must_be_in_subdir = True
        for path, subdirs, files in os.walk(self.data_dir):
            for sdir in subdirs:
                tempdirs.append(sdir)
            for file in files:
                streams.append(file)
        if not tempdirs:
            tempdirs = [self.data_dir.split("/")[-1]]
            file_must_be_in_subdir = False
        if not file_must_be_in_subdir:
            self.file_list[tempdirs[0]] = [
                (file.split("/")[-1].split(".")[0], os.path.join(self.data_dir, x))
                for x in streams
            ]
        else:
            print("sorting files by subdir")
            for file in tqdm(streams):
                file_id = file.split("/")[-1].split(".")[0]
                for sdir in tempdirs:
                    fp = os.path.join(self.data_dir, sdir, file)
                    if os.path.isfile(fp):
                        self.file_list[sdir].append((file_id, fp))
        for k, v in self.file_list.items():
            print(f"split {k}: {len(v)} num images")

    def __call__(self):
        dsets = {}
        num_examples = 0
        num_bytes = 0
        streams = []
        for subdir in tqdm(self.file_list.keys()):
            temp = BytesIO()
            streams.append(temp)
            # open temp file
            with OSFile(temp, "wb") as s:
                # create arrow_writer
                writer = ArrowWriter(features=self.features, stream=s)
                # loop through files in split
                for (img_id, filepath) in tqdm(self.file_list[subdir]):
                    # we assume images will be read in brg format
                    image, sizes, scale_hw = image.img_to_tensor(
                        filepath,
                        min_size=self.model.config.input.min_size_test,
                        max_size=self.model.config.input.max_size_test,
                        mean=self.model.config.model.pixel_mean,
                        sdev=self.model.config.model.pixel_std,
                    )
                    if image is None:
                        self.skipped_ids.add(img_id)
                        continue

                    if torch.cuda.is_available():
                        image, sizes, scale_hw = (
                            image.to(self.device),
                            sizes.to(self.device),
                            scale_hw.to(self.device),
                        )
                    output_dict = self.model(
                        image,
                        sizes,
                        scales_yx=scale_hw,
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
                    break
                num_ex, num_b = writer.finalize()
                num_examples += num_ex
                num_bytes += num_b

            with OSFile(temp, "rb") as s:
                dsets[subdir] = datasets.Dataset.from_buffer(
                    s.read(), split=Split(subdir)
                )
            os.remove(temp)

        # remove temp files, optionally combine split, write to final file
        if len(dsets.values()) == 1:
            final = next(iter(dsets.values()))
        else:
            final = datasets.concatenate_datasets(
                dsets=list(dsets.values()), split=[Split(subdir) for subdir in dsets]
            )

        final = pickle.loads(pickle.dumps(final))
        writer = ArrowWriter(path=self.output_file)
        writer.write_table(final._data)
        e, b = writer.finalize()
        assert e == num_examples and b == num_bytes
        print(f"Success! You wrote {num_examples} entry(s) and {num_bytes >> 20} mb")
        print(f"saved {self.output_file}")
        print(f"num ids skipped: {len(self.skipped_ids)}")
