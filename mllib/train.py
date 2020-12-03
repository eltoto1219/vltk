import datetime
import os
from dataclasses import dataclass

import torch
# from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AdamW  # LxmertForQuestionAnswering,

from .dataloader import BaseLoader
from .evaluate import Evaluator


def log_append(log_file, log_str):
    with open(log_file, "a") as f:
        date = datetime.datetime.now()
        f.write(f"Time: {date} \n")
        f.write(log_str)
        f.flush()


class TrainOutputs:
    accuracy: int

    def __init__(self, accuracy):
        self.accuracy = accuracy


class Trainer:
    def __init__(
        self,
        model,
        dataset_name,
        dataset_config,
        train_config,
        global_config,
        pathes_config,
        loader_config,
        model_config,
    ):

        self.model = model
        self.train_config = train_config
        self.dataset_config = dataset_config
        self.dry_run = train_config.dry_run

        self.loader = BaseLoader(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            loader_config=loader_config,
            global_config=global_config,
            pathes_config=pathes_config,
            train_config=train_config,
            split=dataset_config.split,
        )

        self.log = True
        self.dataset = self.loader.dataset
        self.scheduler = None
        self.seed = global_config.seed
        self.set_seed()
        self.optim = AdamW(list(self.model.parameters()), 1e-04)
        self.log_file = os.path.join(global_config.log_dir, global_config.log_file)

        if not os.path.isfile(self.log_file):
            if not os.path.isdir(global_config.log_dir):
                os.makedirs(global_config.log_dir, exist_ok=True)
            log = open(self.log_file, "w")
            log.close()

        if not dataset_config.skip_eval:
            self.evaluate = Evaluator(
                model_instance=self.model,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                train_config=train_config,
                global_config=global_config,
                pathes_config=pathes_config,
                loader_config=loader_config,
                model_config=model_config,
                split="eval",
            )

        if torch.cuda.is_available() and global_config.gpus != -1:
            self.model.cuda()

    def set_seed(self):
        assert hasattr(self, "seed")
        assert isinstance(self.seed, int)
        torch.manual_seed(self.seed)

    def __call__(self):
        for epoch in range(self.train_config.epochs):
            self.epoch = epoch
            train_outputs = self.train()
            train_acc = train_outputs.accuracy
            if not self.dataset_config.skip_eval:
                val_outputs = self.evaluate()
                val_acc = val_outputs.accuracy

            # scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            # log
            if self.log:
                log_str = (
                    f"\tEpoch {self.epoch}: {self.split}={train_acc:0.2f} % {self.eval_split}"
                    f" {(val_acc):0.2f}%\n"
                )
                if not self.dry_run:
                    log_append(self.log_file, log_str)
                print(log_str)
            # check if test
            if self.dry_run:
                break

    def train(self):
        total = 0.0
        right = 0.0
        for batch in tqdm(self.loader):
            self.model.train()
            self.optim.zero_grad()
            batch = self.loader.toCuda(batch)
            output = self.model(
                input_ids=batch["input_ids"],
                visual_feats=batch["roi_features"],
                visual_pos=batch["boxes"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
                token_type_ids=batch["token_type_ids"],
                return_dict=True,
            )

            # do loss now
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optim.step()

            # find score now
            score = output.question_answering_score
            logit = torch.nn.functional.softmax(score, dim=-1)
            logit, pred = logit.max(1)
            gold = batch["label"]
            right += (gold.eq(pred.long())).sum()
            total += float(self.train_config.train_batch_size)

            # check if test run:
            if self.dry_run:
                break

        return TrainOutputs(accuracy=(right / total) * 100)
