import os

from transformers import BertTokenizerFast, RobertaTokenizerFast
from vltk import build
from vltk.adapters import Adapters
from vltk.configs import DataConfig, LangConfig, VisionConfig

if __name__ == "__main__":
    datadir = os.path.join(os.environ["HOME"], "demodata")
    # Adapters().get("funsd").extract(datadir)
    config = DataConfig(
        # lang=LangConfig(tokenizer="BertWordPieceTokenizer"),
        train_datasets=[["funsd", "train"]],
        visn=VisionConfig(grayscale=True),
        lang=LangConfig(max_visual_seq_length=512, tokenizer=BertTokenizerFast),
        visn_processors=["auxtokenize", "ocrboxfixed", "tokenlabels", "xywhtoxyxy"],
        num_workers=1,
        extractor=None,
        datadir=datadir,
        train_batch_size=2,
        eval_batch_size=2,
        add_visual_cls=True,
        ignore_image=True,
        reextract=True,
        redownload=False,
    )

    train_loader, val_loader = build(config)
    for x in train_loader:
        pass
