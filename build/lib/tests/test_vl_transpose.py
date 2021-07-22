from transformers import RobertaTokenizerFast
from vltk.adapters import Adapters
from vltk.configs import DataConfig, LangConfig
from vltk.dataset.builder import init_datasets
from vltk.processing import LangProcessor


class TestProcessor(LangProcessor):
    def forward(self, x, *args, **kwargs):
        return x


if __name__ == "__main__":

    datadir = "/home/eltoto/demodata"
    config = DataConfig(
        train_datasets=[
            ["gqa", "train"],
            ["vqa", "trainval"],
            ["cococaptions", "trainval"],
            ["vgqa", "train"],
        ],
        extractor="frcnn",
        datadir=datadir,
        num_workers=0,
        train_batch_size=2,
        img_first=True,
        ignore_segmentation=True,
    )
    train, val = init_datasets(config)
    for i, b in enumerate(train):
        if i == 0:
            continue
        b = train.transpose_vl(b)
        break
