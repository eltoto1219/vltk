from transformers import RobertaTokenizerFast
from vltk.adapters import Adapters
from vltk.configs import DataConfig, LangConfig
from vltk.loader import build
from vltk.processing import LangProccessor


class TestProcessor(LangProccessor):
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
        result = Adapter.extract(datadir, dataset="visualgenome")
        result = Adapter.load(datadir)
    if get_loaders_visnlang:
        config = DataConfig(
            lang=LangConfig(
                tokenizer=RobertaTokenizerFast, vocab_path_or_name="roberta-base"
            ),
            processors=[TestProcessor],
            train_datasets=[
                ["vqa", "trainval"],
                # ["gqa", "trainval"],
                # ["vgqa", "train"],
                # ["cococaptions", "trainval"],
            ],
            extractor="frcnn" if use_extractor else None,
            datadir=datadir,
            num_workers=1,
            train_batch_size=1,
            eval_batch_size=1,
            img_first=False,
        )
        train_loader, val_loader = build(config)
        for x in train_loader:
            print(x)
            break
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
