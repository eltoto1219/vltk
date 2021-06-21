from transformers import RobertaTokenizerFast

from vltk.adapters import Adapters
from vltk.configs import DataConfig, LangConfig
from vltk.loader import build
from vltk.processing import LangProccessor


class TestProcessor(LangProccessor):
    def forward(self, x, *args, **kwargs):
        return x


if __name__ == "__main__":
    VGQA = Adapters().get("vgqa")
    datadir = "/home/eltoto/demodata"
    # VGQA.extract(datadir)
    # print(Adapters().avail())
    # superset datasets
    # define config for dataset
    config = DataConfig(
        lang=LangConfig(
            tokenizer=RobertaTokenizerFast, vocab_file_or_name="roberta-base"
        ),
        processors=[TestProcessor],
        # choose which dataset and dataset split for train and eval
        train_datasets=[
            ["vgqa", "train"],
        ],
        # eval_datasets=["gqa", "testdev"],
        # choose which tokenizer to use
        # choose which feature extractor to use
        extractor=None,
        datadir=datadir,
        num_workers=1,
        train_batch_size=1,
        eval_batch_size=1,
        img_first=True,
    )

    train_loader, val_loader = build(config)
    for x in train_loader:
        print(x)
        break
