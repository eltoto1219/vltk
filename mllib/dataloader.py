import torch
from torch.utils.data import DataLoader, Dataset
import datasets
from operator import itemgetter
import json
from transformers import LxmertTokenizer
from pynvml.smi import nvidia_smi
import os
from collections import defaultdict
import timeit
import functools
from tqdm import tqdm


DEV = sorted([(i, d["fb_memory_usage"]["free"]) for i, d in enumerate(nvidia_smi.getInstance().DeviceQuery("memory.free")["gpu"])], key = lambda x: x[0])[-1][0]


def get_duration(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        starttime = timeit.default_timer()
        output = func(*args, **kwargs)
        print(f"exec: {func.__name__} in {timeit.default_timer() - starttime:.3f} s")
        return output
    return wrapper


def univeral_collate(columns):
    batch = {}
    for x in map(lambda x:iter(x.items()), columns):
        for k,v in x:
            if not hasattr(batch,k):
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = torch.cat((batch[k], v.unsqueeze(0)), dim=0)
    return batch


class MMF(Dataset):
    @get_duration
    def __init__(self, text_file, arrow_file, sent_length=20):
        super().__init__()
        self.max_length = sent_length
        self.arrow_dataset = datasets.Dataset.from_file(arrow_file)
        self.id2idx = {k:i for i, k in enumerate(self.arrow_dataset["img_id"])}
        self.arrow_dataset.set_format(type="numpy", output_all_columns=True)
        self.text_file = text_file
        self.text_data = [self.mmf_txt_format(json.loads(i)) for i in open(text_file)]
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

    def mmf_txt_format(self, entry):
        return (int(entry["label"]), entry["text"],int(entry["id"]))

    def mmf_imgid_format(self, filename):
        return int(filename.split(".")[0])

    def __len__(self):
        return len(self.text_data)

    @torch.no_grad()
    def __getitem__(self, i):
        label, text, img_id = self.text_data[i]
        img_id = self.id2idx[img_id]
        img_data = self.arrow_dataset[img_id]
        for k in img_data:
            img_data[k] = torch.from_numpy(img_data[k].copy())
        inputs = self.lxmert_tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        img_data["input_ids"] = inputs.input_ids.squeeze(0)
        img_data["attention_mask"] = inputs.attention_mask.squeeze(0)
        img_data["token_type_ids"] = inputs.token_type_ids.squeeze(0)
        img_data["label"] = torch.tensor(label)
        return img_data


class MMFLoader(DataLoader):
    def __init__(self, text_file, arrow_file, batch_size=128, sent_length=20):
        shuffle = 1 if "test" in text_file else 0
        super().__init__(
            dataset=MMF(text_file, arrow_file),
            collate_fn=univeral_collate,
            drop_last=False,
            pin_memory=True,
            num_workers=8,
            shuffle=shuffle,
            batch_size=batch_size
        )


if __name__ == "__main__":
    loader = MMFLoader("train.jsonl", "mmf.arrow")
    for x in tqdm(loader):
        pass
