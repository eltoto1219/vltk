from collections import OrderedDict

import datasets as ds
from mllib.abc.textset import Textset
from mllib.configs import Config

TESTPATH = "/playpen1/home/avmendoz/data/vqa/train"


# user must only define forward in this function


class VQAset(Textset):
    dataset = "vqa"
    features = ds.Features(OrderedDict({}))

    def train_location(self, config):
        return config.pathes.vqa_train

    def eval_location(self, config):
        return config.pathes.vqa_eval


if __name__ == "__main__":

    config = Config()
    textset = VQAset.extract(
        config=config, path=TESTPATH, superkey="questions", save_to="test.arrow"
    )

    loaded = VQAset.from_file("test.arrow")
    # imap = loaded.img_to_row_map
    # imgid = next(iter(imap.keys()))
    # print(imap)
    # print("is aligned?", loaded.check_imgid_alignment())
    # print(f"entry for {imgid}: ", loaded.get_image(imgid).keys())

    # # load specific split:
    # loaded = FRCNNSet.from_file("test.arrow", split="subdir_extract_2")
    # print(loaded)
