from tqdm import tqdm

from .dataloader import BaseLoader


class Data:
    def __init__(
        self,
        dataset_name,
        dataset_config,
        loader_config,
        global_config,
        pathes_config,
        train_config,
        pretrain_config=None,
    ):

        self.loader = BaseLoader(
            dataset_name,
            dataset_config,
            loader_config,
            global_config,
            pathes_config,
            train_config,
        )

        self.datset = self.loader.dataset

    def full_pass(self):
        for x in tqdm(self.loader):
            pass

    def keys_and_shapes(self):
        entry = None
        for x in self.loader:
            entry = x
            break

        input_ids = entry["input_ids"]
        masked_inds = entry["masked_inds"]
        label = entry["label"]
        is_matched = entry["is_matched"]
        img_id = entry["img_id"]
        sents = []
        for entry in input_ids:
            sents.append(self.dataset.tokenizer.convert_ids_to_tokens(entry))
        print("input_ids", input_ids)
        print("sents", sents)
        print("masked_inds", masked_inds)
        print("img_id", img_id)
        print("is_matched", is_matched)
        print("label", label)
        print("special", self.dataset.special_ids)
