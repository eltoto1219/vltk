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

    def tpass(self):
        for x in tqdm(self.loader):
            pass

    def keys(self):
        entry = None
        for x in self.loader:
            entry = x
            for k, v in entry.items():
                print(k, type(v))
            break
