from transformers import RobertaTokenizerFast

from vltk.adapters import Adapters
from vltk.configs import DataConfig, LangConfig
from vltk.loader import build
from vltk.processing import LangProccessor


class TestProcessor(LangProccessor):
    def forward(self, x, *args, **kwargs):
        return x


if __name__ == "__main__":
    datadir = "/home/eltoto/demodata"
    config = DataConfig(
        lang=LangConfig(
            tokenizer=RobertaTokenizerFast, vocab_file_or_name="roberta-base"
        ),
        processors=[TestProcessor],
        train_datasets=[
            ["vqa", "trainval"],
            ["gqa", "trainval"],
            ["vgqa", "train"],
            ["cococaptions", "trainval"],
        ],
        extractor="frcnn",
        datadir=datadir,
        num_workers=1,
        train_batch_size=5,
        eval_batch_size=1,
        img_first=False,
    )

    train_loader, val_loader = build(config)
    for x in train_loader:
        print(x)
        break
