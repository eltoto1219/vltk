import os
import time
from collections import Counter, defaultdict

from tqdm import tqdm

import vltk
import vltk.Features
from vltk import Features, compat
from vltk.abc.extraction import VizExtractionAdapter, VizExtractionAdapters
from vltk.abc.visnadapter import VisnDatasetAdapter, VisnDatasetAdapters
from vltk.abc.visnlangadatper import (VisnLangDatasetAdapter,
                                      VisnLangDatasetAdapters)
from vltk.configs import DataConfig, ProcessorConfig
from vltk.loader.builder import init_datasets
from vltk.metrics import soft_score
from vltk.modeling.frcnn import FRCNN as FasterRCNN
from vltk.processing.label import clean_imgid_default


class VisualGenome(VisnDatasetAdapter):
    def schema():
        return {}

    def forward(json_files, **kwargs):
        return {}


class GQA(VisnLangDatasetAdapter):
    data_info = {
        "dev": {"coco2014": ["test"]},
        "train": {"visualgenome": ["train"]},
        "val": {"visualgenome": ["train"]},
        "test": {"coco2014": ["test"]},
        "testdev": {"coco2014": ["val"]},
    }
    schema = {}

    def forward(json_files, split, **kwargs):
        skipped = 0
        min_label_frequency = kwargs.get("min_label_frequency", 2)
        label_preprocessor = kwargs.get("label_preprocessor", None)
        label_frequencies = Counter()
        batch_entries = []
        if label_preprocessor is None:

            def label_preprocessor(x):
                return x

        for t in json_files:
            for i, (k, v) in enumerate(t.items()):
                if "answer" in v:
                    answer = label_preprocessor(v["answer"])
                    label_frequencies.update([answer])

            for i, (k, v) in enumerate(t.items()):
                if split == "test":
                    answer = None
                elif label_frequencies[v["answer"]] < min_label_frequency:
                    skipped += 1
                    continue
                else:
                    answer = label_preprocessor(v["answer"])

                text = v["question"]
                img_id = v["imageId"].lstrip("n")
                entry = {
                    vltk.text: text,
                    vltk.imgid: img_id,
                    vltk.label: [answer],
                    vltk.score: [1.0],
                }

                batch_entries.append(entry)

        print(f"SKIPPEd {skipped} entries")
        return batch_entries


if __name__ == "__main__":
    datadir = "/ssd-playpen/avmendoz/data"
    Vizlang = VisnLangDatasetAdapters()
    Viz = VisnDatasetAdapters()
    Extract = VizExtractionAdapters()
    # gqa = GQA.extract(datadir)
    # visualgenome = VisualGenome.extract(datadir)
    Vizlang.add(GQA)
    Viz.add(VisualGenome)

    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        config = DataConfig(
            train_datasets=["gqa", "train"],
            eval_datasets=None,
            tokenizer="BertWordPeice",
            extractor=None,
            datadir=datadir,
            train_batch_size=8,
            img_first=False,
            percent=p,
            num_workers=8,
        )
        # use config to create dataset
        (train, val), _, answer_to_id, object_to_id = init_datasets(config)
        print("NUM EXAMPLES", len(train[1]))
        start = time.time()
        for x in tqdm(train[1]):
            pass
        stop = time.time()
        print(stop - start)
