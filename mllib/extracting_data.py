import getopt
import json
import os
from pynvml.smi import nvidia_smi

# import numpy as np
import sys
from collections import OrderedDict
from subprocess import Popen, PIPE
from tqdm import tqdm

import datasets
import numpy as np
import torch
from pynvml.smi import nvidia_smi

from mllib import Config, GeneralizedRCNN, Preprocess


"""
USAGE:
``python extracting_data.py -i <img_dir> -o <dataset_file>.datasets <batch_size>``
"""

#consider one image -> many questions, or one image one questions
#will need to test the effectiveness of grouping by questions

DEV = (
    sorted(
        [
            (i, d["fb_memory_usage"]["free"])
            for i, d in enumerate(
                nvidia_smi.getInstance().DeviceQuery("memory.free")["gpu"]
            )
        ],
        key=lambda x: x[0],
    )[-1][0]
    if torch.cuda.is_available()
    else -1
)

DEV = f"cuda:{0}" if DEV > 0 else "cpu"
print(DEV)



TEST = False
CONFIG = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
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
            "img_id": datasets.Value("int32"),
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


class Extract:
    def __init__(self, argv=sys.argv[1:]):
        inputdir = None
        outputfile = None
        subset_list = None
        y_ignore = None
        batch_size = 1
        opts, args = getopt.getopt(
                argv, "i:o:y:b", ["inputdir=", "outfile=", "y_ignore=", "batch_size="]
        )
        for opt, arg in opts:
            if opt in ("-i", "--inputdir"):
                inputdir = arg
            elif opt in ("-o", "--outfile"):
                outputfile = arg
            elif opt in ("-y", "--yignore"):
                y_ignore = arg
            elif opt in ("-b", "--batch_size"):
                batch_size = int(arg)

        assert inputdir is not None  # and os.path.isdir(inputdir), f"{inputdir}"
        assert outputfile is not None and not os.path.isfile(
            outputfile
        ), f"{outputfile}"
        if subset_list is not None:
            with open(os.path.realpath(subset_list)) as f:
                self.subset_list = set(
                    map(lambda x: self._vqa_file_split()[0], tryload(f))
                )
        else:
            self.subset_list = None

        self.config = CONFIG
        self.y_ignore = None if y_ignore is None else json.load(open(y_ignore))
        # EDIT SETTINGS HERE
        # self.config.roi_heads.nms_thresh_test = [0.2]
        # self.config.roi_heads.score_thresh_test = 0.1
        # self.config.min_detections = 0
        # self.config.max_detections = 36
        self.inputdir = os.path.realpath(inputdir)
        self.outputfile = os.path.realpath(outputfile)
        self.num = int(Popen(f"ls {self.inputdir}|wc -l", shell=True, stdout=PIPE).communicate()[0])
        self.preprocess = Preprocess(self.config)
        self.model = GeneralizedRCNN.from_pretrained(
            "unc-nlp/frcnn-vg-finetuned", config=self.config
        )
        if torch.cuda.is_available():
            self.dev_id = DEV
            self.model.to(self.dev_id)
        self.batch = batch_size if batch_size != 0 else 1
        self.schema = DEFAULT_SCHEMA
        self.progress = tqdm(unit_scale=True, total=self.num, desc="Extracting")

    def _vqa_file_split(self, file):
        img_id = int(file.split(".")[0].split("_")[-1])
        filepath = os.path.join(self.inputdir, file)
        return (img_id, filepath)

    @property
    def file_generator(self):
        batch = []
        for i, file in enumerate(os.listdir(self.inputdir)):
            if self.subset_list is not None and i not in self.subset_list:
                continue
            batch.append(self._vqa_file_split(file))
            if len(batch) == self.batch:
                temp = batch
                batch = []
                yield list(map(list, zip(*temp)))

        for i in range(1):
            yield list(map(list, zip(*batch)))

    def __call__(self):
        # make writer
        if not TEST:
            writer = datasets.ArrowWriter(features=self.schema, path=self.outputfile)
        # do file generator
        for i, (img_ids, filepaths) in enumerate(self.file_generator):
            ignorey=None
            if self.y_ignore is not None:
                ignorey = []
                for i in img_ids:
                    ignorey.append(torch.Tensor(self.y_ingnore[i]))
            images, sizes, scales_yx = self.preprocess(filepaths)
            if torch.cuda.is_available():
                images, sizes, scales_yx = (
                    images.to(self.dev_id),sizes.to(self.dev_id),scales_yx.to(self.dev_id)
                )
            output_dict = self.model(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=self.config.MAX_DETECTIONS,
                pad_value=0,
                return_tensors="np",
                location="cpu",
                ignorey=ignorey
            )
            output_dict["boxes"] = output_dict.pop("normalized_boxes")
            if not TEST:
                output_dict["img_id"] = np.array(img_ids)
                batch = self.schema.encode_batch(output_dict)
                writer.write_batch(batch)
                self.progress.update(self.batch)
            else:
                break
            # finalizer the writer
        if not TEST:
            num_examples, num_bytes = writer.finalize()
            print(f"Success! You wrote {num_examples} entry(s) and {num_bytes >> 20} mb")


def tryload(stream):
    try:
        data = json.load(stream)
        try:
            data = list(data.keys())
        except Exception:
            data = [d["img_id"] for d in data]
    except Exception:
        try:
            data = eval(stream.read())
        except Exception:
            data = stream.read().split("\n")
    return data


if __name__ == "__main__":
    extract = Extract(sys.argv[1:])
    extract()
    if not TEST:
        dataset = datasets.Dataset.from_file(extract.outputfile)
        # wala!
        #print(np.array(dataset[0:2]["roi_features"]).shape)
