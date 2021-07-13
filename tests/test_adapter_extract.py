from transformers import RobertaTokenizerFast
from vltk import build
from vltk.adapters import Adapters
from vltk.configs import DataConfig, LangConfig
from vltk.processing import LangProcessor


class TestProcessor(LangProcessor):
    def forward(self, x, *args, **kwargs):
        return x


if __name__ == "__main__":
    get_loaders_visnlang = True
    get_loaders_visn = False
    extract_visn = False
    extract_visnlang = False
    extract_extractor = False
    datadir = "/home/eltoto/demodata"
    use_extractor = False
    if extract_visnlang:
        Adapter = Adapters().get("vqa")
        result = Adapter.extract(datadir)["val"]
        result = Adapter.load(datadir)["val"]
    if extract_visn:
        Adapter = Adapters().get("coco2014")
        result = Adapter.extract(datadir)
        result = Adapter.load(datadir)
    if extract_extractor:
        Adapter = Adapters().get("frcnn")
        result = Adapter.extract(datadir, dataset="coco2014")
        result = Adapter.load(datadir)
    if get_loaders_visnlang:
        # add adapters to library

        # config
        config = DataConfig(
            # choose which dataset and dataset split for train and eval
            train_datasets=[
                ["gqa", "train"],
                ["vqa", "trainval"],
                ["cococaptions", "trainval"],
                ["vgqa", "train"],
            ],
            # choose which feature extractor to use
            extractor="frcnn",
            datadir=datadir,
            num_workers=8,
            train_batch_size=1,
            # iterate with through datasets via images first versus text
            img_first=True,
            # ignore segmentation annotations from being prcoessed with the COCO dataset
            ignore_segmentation=True,
        )
        train, val = init_datasets(config)
        for b in train:
            print(b.keys())
            pass

    if get_loaders_visn:

        config = DataConfig(
            train_datasets=[
                ["coco2014", "train"],
            ],
            datadir=datadir,
            num_workers=1,
            train_batch_size=1,
            eval_batch_size=1,
        )

        train_loader, val_loader = build(config)
        for x in train_loader:
            print(x)
            break
