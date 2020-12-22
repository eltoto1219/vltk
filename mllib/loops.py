import sys
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from mllib import data, metrics, utils
from mllib.outputs import LoopOutputs


class BaseLoop(ABC):
    def __init__(self, config, datasets, model_dict, extra_modules=None):
        self.model_dict = model_dict
        self.extra_modules = extra_modules
        self.config = config
        self.datasets = datasets
        self.cur_step = 0
        self.scheduler = None
        self.half_precision = getattr(config.run, "half_precision", False)
        self.dryrun = getattr(config, "dryrun", False)
        self.main_device = f"cuda:{config.gpu}"\
            if getattr(config, "gpu", -1) != -1 else "cpu"
        self.aux_model_device = f"cuda:{config.aux_gpu}"\
            if getattr(config, "aux_gpu", -1) != -1 else "cpu"
        self.scaler = None if not self.half_precision else torch.cuda.amp.GradScaler()
        self._init_loader()
        assert hasattr(self, "loader"), "property 'loader' must be set in 'self._init_loader()'"
        assert isinstance(self.loader, torch.utils.data.DataLoader)
        self._dataset = self.loader.dataset
        self._init_models_and_extras(model_dict, extra_modules)
        self._init_optim()

    def _init_models_and_extras(self, model_dict, extra_modules=None):
        for k, v in model_dict.items():
            v_old = getattr(self, k, None)
            assert v_old is None, (type(v_old), k)
            setattr(self, k, v)
        if extra_modules is not None:
            for k, v in extra_modules.items():
                v_old = getattr(self, k, None)
                assert v_old is None, type(v_old)
                v = v.to(self.main_device)
                setattr(self, k, v)

    def _init_optim(self):
        if self.is_train:
            parameters = []
            for k, v in self.model_dict.items():
                parameters.extend(v.parameters())
            if self.extra_modules is not None:
                for k, v in self.extra_modules.items():
                    parameters.extend(v.parameters())
            assert parameters, "no parameters added to optimizer"
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

    @property
    def forward_context(self):
        if self.scaler is None:
            return utils.dummy_context
        else:
            return torch.cuda.amp.autocast

    @property
    def tqdm(self):
        desc = "train" if self.is_train else "eval"
        self._tqdm = tqdm(self.loader, desc=desc, ncols=0, file=sys.stdout)
        return self._tqdm

    @property
    def batch_size(self):
        if getattr(self, "_bz", None) is not None:
            return self._bz
        else:
            if self.is_train:
                return self.config.run.batch_size
            else:
                return self.config.evaluate.batch_size

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
    def dataset(self):
        return self._dataset

    def set_batch_size(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                self._bz = v.size(0)
                break

    def get_grad_params(self):
        parameters = []
        for k, v in self.model_dict.items():
            parameters.extend([p for p in v.parameters() if p.requires_grad])
        if self.extra_modules is not None:
            for k, v in self.extra_modules.items():
                parameters.extend([p for p in v.parameters() if p.requires_grad])
        return parameters

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

    def step(self, loss=None):
        if self.is_train and loss is not None:
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    self.get_grad_params(), self.config.run.max_norm
                )
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.get_grad_params(), self.config.run.max_norm
                )
                self.optim.step()

            self.warmup.step()

    def get_lr(self):
        if not self.is_train:
            return []
        elif self.warmup is not None:
            return self.warmup.get_lr()
        else:
            lrs = []
            for param_group in self.optim.param_groups:
                lrs.append(param_group['lr'])
            return lrs

    def __iter__(self):
        for x in self.loader:
            yield x

    def __call__(self) -> LoopOutputs:
        accuracy = 0.0
        losses = None
        if self.is_train:
            self.toTrain()
        else:
            self.toEval()
        with torch.no_grad() if not self.is_train else utils.dummy_context():
            for batch in self.tqdm:
                if self.is_train:
                    self.optim.zero_grad()
                self.set_batch_size(batch)
                with self.forward_context():
                    model_outputs = self.forward(batch)
                outputs = self.loop(batch, model_outputs)
                if outputs is not None:
                    if hasattr(outputs, "losses"):
                        losses = outputs.losses
                    if hasattr(outputs, "accuracy"):
                        accuracy += float(outputs.accuracy)
                if self.is_train:
                    losses = getattr(outputs, "losses", None)
                    assert losses is not None
                    self.step(losses)
                self.cur_step += 1
                if self.config.dryrun:
                    break
            outputs = LoopOutputs(right=accuracy, total=len(self.loader), losses=losses)
            return outputs

    @abstractmethod
    def loop(self, batch, model_outputs) -> LoopOutputs:
        return None

    @abstractmethod
    def forward(self, batch) -> object:
        return None

    def _init_loader(self):
        if self.is_train:
            split = "train"
        else:
            split = self.config.data.eval_split
        self.loader = data.UniversalLoader(
            config=self.config,
            split=split,
            dataset_name=self.datasets
        )

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def is_train(self):
        pass


class Data(BaseLoop):
    name: str = "data"
    is_train: bool = False

    def forward(self, batch):
        return super().forward(batch)

    def loop(self, batch, model_outputs):
        return super().loop(batch, model_outputs)


class EvalLxmert(BaseLoop):
    name: str = "evallxmert"
    is_train: bool = False

    def loop(self, batch, model_outputs):
        acc = metrics.accuracy(model_outputs.question_answering_score, batch["labels"])
        loop_outputs = LoopOutputs(accuracy=acc)
        self.tqdm_update({"acc": loop_outputs.accuracy})
        return loop_outputs

    def forward(self, batch):
        self.toCuda(batch, device=self.config.gpu)
        batch["return_dict"] = True
        model_outputs = self.lxmert(
            input_ids=batch["input_ids"],
            visual_feats=batch["roi_features"],
            visual_pos=batch["boxes"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            return_dict=batch["return_dict"],
        )
        return model_outputs


class TrainLxmert(BaseLoop):
    name: str = "trainlxmert"
    is_train: bool = True

    def loop(self, batch, model_outputs):
        acc = metrics.accuracy(model_outputs.question_answering_score, batch["labels"])
        loop_outputs = LoopOutputs(accuracy=acc, losses=model_outputs.loss)
        self.tqdm_update(
            {
                "acc": f"{acc:.3f}%",
                "lrs": [f"{l:.3e}" for l in self.get_lr()],
                "loss": f"{model_outputs.loss:.3f}",
            }
        )
        return loop_outputs

    def forward(self, batch):
        self.toCuda(batch, device=self.config.gpu)
        batch["return_dict"] = True
        model_outputs = self.lxmert(
            input_ids=batch["input_ids"],
            visual_feats=batch["roi_features"],
            visual_pos=batch["boxes"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            return_dict=batch["return_dict"],
            labels=batch["labels"],
        )
        return model_outputs


class EvalViTLxmert(BaseLoop):
    name: str = "evalvitlxmert"
    is_train: bool = False

    def loop(self, batch, model_outputs):
        acc = metrics.accuracy(model_outputs.question_answering_score, batch["labels"])
        loop_outputs = LoopOutputs(accuracy=acc)
        self.tqdm_update({"acc": loop_outputs.accuracy})
        return loop_outputs

    def forward(self, batch):
        assert self.config.data.use_raw_imgs and self.config.data.img_first, "must set aformentioned options"
        self.toCuda(batch, device=self.aux_model_device)
        split_imgs = torch.split(batch.pop("raw_imgs"), split_size_or_sections=1, dim=0)
        combined_feats = None
        for split in split_imgs:
            feats = self.vit(split).to(self.main_device)
            if combined_feats is None:
                combined_feats = feats.to(self.main_device)
            else:
                combined_feats = torch.cat(
                    (combined_feats, feats.to(self.main_device)), dim=0
                )
        shape = combined_feats.shape
        combined_feats = combined_feats.view(shape[0], int(shape[1] ** 0.5), -1)
        batch["roi_features"] = self.connector(combined_feats)
        batch["boxes"] = torch.zeros((shape[0], int(shape[1] ** 0.5), 4)).to(
            self.main_device
        )
        # print()
        # print("---")
        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape)

        self.dataset.transpose_img2txt(batch, img_keys=["boxes", "roi_features"])
        self.toCuda(batch, device=self.main_device)
        batch["return_dict"] = True
        model_outputs = self.lxmert(
            input_ids=batch["input_ids"],
            visual_feats=batch["roi_features"],
            visual_pos=batch["boxes"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            return_dict=batch["return_dict"],
        )
        return model_outputs


class TrainViTLxmert(BaseLoop):
    name: str = "trainvitlxmert"
    is_train: bool = True

    def loop(self, batch, model_outputs):
        acc = metrics.accuracy(model_outputs.question_answering_score, batch["labels"])
        loop_outputs = LoopOutputs(accuracy=acc, losses=model_outputs.loss)
        flatten_bz = len(batch["input_ids"])
        self.tqdm_update(
            {
                "acc": f"{acc:.3f}%",
                "lrs": [f"{l:.3e}" for l in self.get_lr()],
                "loss": f"{model_outputs.loss:.3f}",
                "bz": flatten_bz
            }
        )
        return loop_outputs

    def forward(self, batch):
        assert self.config.data.use_raw_imgs and self.config.data.img_first, "must set aformentioned options"
        self.toCuda(batch, device=self.aux_model_device)
        split_imgs = torch.split(batch.pop("raw_imgs"), split_size_or_sections=1, dim=0)
        combined_feats = None
        for split in split_imgs:
            feats = self.vit(split).to(self.main_device)
            if combined_feats is None:
                combined_feats = feats.to(self.main_device)
            else:
                combined_feats = torch.cat(
                    (combined_feats, feats.to(self.main_device)), dim=0
                )
        shape = combined_feats.shape
        combined_feats = combined_feats.view(shape[0], int(shape[1] ** 0.5), -1)
        batch["roi_features"] = self.connector(combined_feats)
        batch["boxes"] = torch.zeros((shape[0], int(shape[1] ** 0.5), 4)).to(
            self.main_device
        )

        # for k, v in batch.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape)

        self.dataset.transpose_img2txt(batch, img_keys=["boxes", "roi_features"])

        self.toCuda(batch, device=self.main_device)
        batch["return_dict"] = True
        model_outputs = self.lxmert(
            input_ids=batch["input_ids"],
            visual_feats=batch["roi_features"],
            visual_pos=batch["boxes"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            return_dict=batch["return_dict"],
            labels=batch["labels"]
        )
        return model_outputs
