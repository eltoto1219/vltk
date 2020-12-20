import datetime
import os
from abc import ABC, abstractmethod
from typing import List, Union

import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from mllib import data, metrics, utils
from mllib.models import factories


class Loop(ABC):
    def __init__(self, config, model=None, aux_model=None):
        assert hasattr(self, "_loader") and hasattr(self, "_run"), (
            "must call `self.set_loader(loader)` and"
            "`self.set_run(run)` before calling `super` to Abstract class"
        )
        self.cur_step = 0
        self.model = model
        self.aux_model = aux_model
        self.config = config
        self.seed = config.seed
        self.half_precision = getattr(config.run, "half_precision", False)
        self.main_device = f"cuda:{config.gpu}" if config.gpu != -1 else "cpu"
        self.aux_model_device = f"cuda:{config.aux_gpu}" if config.gpu != -1 else "cpu"
        if self.model is not None and self.model.device != self.main_device:
            self.model.to(self.main_device)
        if self.aux_model is not None:
            self.aux_model.to(self.aux_model_device)
            self.connector = torch.nn.Linear(18464, 2048)
            self.connector.to(self.main_device)
        self.scheduler = None
        self.dryrun = config.dryrun
        self.scaler = (
            None
            if not self.half_precision or self.run == "data"
            else torch.cuda.amp.GradScaler()
        )
        if self.loader.drop_last:
            print(
                "WARNING: drop last set in loader, so batch size may not be the same for last batch"
            )
            print()

    @property
    def run(self):
        return self._run

    @property
    def tqdm(self):
        assert hasattr(self, "_run"), "must call :set_run(run) before calling tqdm"
        desc = "train" if self.is_train else "eval"
        self._tqdm = tqdm(self.loader, desc=desc, ncols=0)
        return self._tqdm

    @property
    def batch_size(self):
        if self.is_train:
            return self.config.run.batch_size
        else:
            return self.config.run.batch_size

    @property
    def total_steps(self):
        if self.is_train:
            return self.config.run.epochs * len(self.loader)
        else:
            return len(self.loader)

    @property
    def warmup(self):
        return getattr(self, "_warmup", None)

    @property
    def optim(self):
        return getattr(self, "_optim", None)

    @property
    def is_train(self):
        if self.run == "train":
            return True
        else:
            return False

    @property
    def loader(self):
        return self._loader

    @property
    def dataset(self):
        return self._dataset

    def set_run(self, run):
        valid_runs = ("eval", "train", "data")
        assert run in valid_runs, f"run {run} must be a valid run ({valid_runs})"
        self._run = run

    def set_loader(self, loader):
        assert isinstance(loader, torch.utils.data.DataLoader)
        self._loader = loader
        self._dataset = loader.dataset

    def tqdm_update(self, info: dict = None):
        if info is not None:
            self._tqdm.set_postfix(**info)

    def toCuda(self, batch, device=None):
        self.loader.toCuda(batch, device)

    def toTrain(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                v.train()

    def toEval(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                v.eval()

    def run_vit(self, batch, accum_iters=1):
        imgs = batch["raw_imgs"]
        if self.config.data.img_first:
            split_imgs = imgs
        else:
            split_imgs = torch.split(imgs, split_size_or_sections=accum_iters, dim=0)
        combined_feats = None
        for split in split_imgs:
            feats = self.aux_model(split)
            if combined_feats is None:
                combined_feats = feats.to(self.main_device)
            else:
                combined_feats = torch.cat(
                    (combined_feats, feats.to(self.main_device)), dim=0
                )
        shape = combined_feats.shape
        combined_feats = combined_feats.view(shape[0], int(shape[1] ** 0.5), -1)
        combined_feats = self.connector(combined_feats)
        batch["boxes"] = torch.zeros((shape[0], int(shape[1] ** 0.5), 4)).to(
            self.main_device
        )
        if self.config.data.img_first:
            n_repeats = len(batch["input_ids"])
            batch["roi_features"] = combined_feats.cuda(self.main_device).repeat(
                n_repeats, 1, 1
            )
        else:
            batch["roi_features"] = combined_feats.cuda(self.main_device)

    def forward(self, batch):
        if self.model is not None:
            if self.is_train:
                self.optim.zero_grad()
            with torch.cuda.amp.autocast() if getattr(
                self, "scaler", None
            ) is not None else utils.dummy_context():
                if self.config.data.use_raw_imgs:
                    assert self.aux_model is not None
                    self.toCuda(batch, device=self.aux_model_device)
                    self.run_vit(batch)
                self.toCuda(batch, device=self.config.gpu)
                batch["return_dict"] = True
                model_outputs = self.model(
                    input_ids=batch["input_ids"],
                    visual_feats=batch["roi_features"],
                    visual_pos=batch["boxes"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    return_dict=batch["return_dict"],
                    labels=batch["labels"],
                )
            return model_outputs

    def __call__(self) -> LoopOutputs:
        accuracy = 0.0
        losses = None
        if self.is_train:
            self.toTrain()
        else:
            self.toEval()
        for batch in self.tqdm:
            outputs = self.loop(batch)
            if outputs is not None:
                if hasattr(outputs, "losses"):
                    losses = outputs.losses
                if hasattr(outputs, "accuracy"):
                    accuracy += outputs.accuracy
            self.cur_step += 1
            if self.config.dryrun:
                break
        outputs = LoopOutputs(right=accuracy, total=len(self.loader), losses=losses)
        return outputs

    @abstractmethod
    def loop(self, batch) -> LoopOutputs:
        pass


class Data(Loop):
    def __init__(self, datasets, config):
        self.set_run("data")
        self.set_loader(data.UniversalLoader(datasets, config, split=config.data.split))
        super().__init__(config=config)

    def loop(self, batch):
        pass

    def keys(self):
        entry = None
        for x in self.loader:
            entry = x
            for k, v in entry.items():
                print(k, type(v))
            break

    def transpose(self):
        assert self.config.data.img_first
        assert not self.config.data.arrow_fields
        for x in self.loader:
            entry = x
            print("BEFORE")
            for k, v in entry.items():
                shape = None
                if isinstance(v, torch.Tensor):
                    shape = v.shape
                print(k, type(v), shape)
            print("AFTER")
            data.UniversalDataset.transpose_img2txt(entry, img_keys=["raw_imgs", "raw_sizes"], device="cpu")
            for k, v in entry.items():
                shape = None
                if isinstance(v, torch.Tensor):
                    shape = v.shape
                print(k, type(v), shape)
            break


class Evaluate(Loop):
    def __init__(self, dataset_names, config, model, aux_model=None):
        self.set_run("eval")
        self.set_loader(data.UniversalLoader(config=config, split=config.data.eval_split))
        super().__init__(model=model, config=config, aux_model=aux_model)

    @torch.no_grad()
    def loop(self, batch):
        output = self.forward(batch)
        acc = metrics.accuracy(output.question_answering_score, batch["labels"])
        loop_outputs = LoopOutputs(accuracy=acc)
        self.tqdm_update({"acc": loop_outputs.accuracy})
        return loop_outputs


class Train(Loop):
    def __init__(self, dataset_names, config, model, aux_model=None):
        self.set_run("train")
        self.set_loader(data.UniversalLoader(dataset_names, config))
        super().__init__(model=model, config=config, aux_model=aux_model)
        self.set_optim()

    def set_optim(self):
        parameters = []
        print_list = ""
        for k, v in self.__dict__.items():
            if isinstance(v, torch.nn.Module):
                params = []
                for n, p in v.named_parameters():
                    if p.requires_grad:
                        print_list += f"{(k, n)} of shape {p.shape}\n"
                        params.append(p)
                parameters.extend(params)
        self._optim = AdamW(
            parameters,
            lr=self.config.run.learning_rate,
            weight_decay=self.config.run.weight_decay,
        )
        if self.config.run.warmup == 0.0:
            self._warmup = None
        total = self.total_steps
        n_steps = int(total * self.config.run.warmup)
        self._warmup = get_linear_schedule_with_warmup(
            self._optim, num_warmup_steps=n_steps, num_training_steps=total
        )

    def step(self, loss):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.run.max_norm
            )
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.run.max_norm
            )
            self.optim.step()

        self.warmup.step()

    def loop(self, batch):
        model_output = self.forward(batch)
        acc = metrics.accuracy(model_output.question_answering_score, batch["labels"])
        loop_outputs = LoopOutputs(accuracy=acc, losses=model_output.loss)
        self.step(model_output.loss)
        self.tqdm_update(
            {
                "acc": f"{acc:.3f}%",
                "lrs": [f"{l:.3e}" for l in self.warmup.get_lr()],
                "loss": f"{model_output.loss:.3f}",
            }
        )
        return loop_outputs
