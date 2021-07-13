import os

from transformers import RobertaTokenizerFast
from vltk.adapters import Adapters
from vltk.configs import DataConfig, LangConfig, VisionConfig
from vltk import build

if __name__ == "__main__":
    datadir = os.path.join(os.environ["HOME"], "demodata")
    Adapters().get("docvqa").extract(datadir)
    # Adapters().get("docvqavisn").extract(datadir)
    # config = DataConfig(
    #     # lang=LangConfig(tokenizer="BertWordPieceTokenizer"),
    #     lang=LangConfig(
    #         tokenizer=RobertaTokenizerFast, vocab_path_or_name="roberta-base"
    #     ),
    #     visn=VisionConfig(grayscale=True),
    #     train_datasets=[["docvqa", "trainval"]],
    #     visn_processors=["auxtokenize", "ocrbox"],
    #     visnlang_processors=["span"],
    #     extractor=None,
    #     datadir=datadir,
    #     train_batch_size=2,
    #     eval_batch_size=2,
    #     img_first=True,
    #     num_workers=1,
    # )

    # train_loader, val_loader = build(config)
    # for x in train_loader:
    #     print(x)
    #     break
