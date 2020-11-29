import datetime
import os

import torch
# from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AdamW  # LxmertForQuestionAnswering,

from .dataloader import BaseLoader
from .mapping import NAME2MODEL


def log_append(log_file, log_str):
    with open(log_file, "a") as f:
        date = datetime.datetime.now()
        f.write(f"Time: {date} \n")
        f.write(log_str)
        f.flush()


class Trainer:
    def __init__(
        self,
        model_name,
        dataset_name,
        dataset_config,
        train_config,
        global_config,
        pathes_config,
        loader_config,
        model_config,
    ):

        self.log_file = os.path.join(global_config.log_dir, global_config.log_file)
        if not os.path.isfile(self.log_file):
            os.mknod(self.log_file)

        self.model = NAME2MODEL[model_name](
            model_name=model_name,
            command_name="train",
            dataset_config=dataset_config,
            run_config=train_config,
        )

        self.loader = BaseLoader(
            dataset_name,
            dataset_config,
            loader_config,
            global_config,
            pathes_config,
            train_config,
        )

        self.inference_loader = BaseLoader(
            dataset_name,
            dataset_config,
            loader_config,
            global_config,
            pathes_config,
            train_config,
            split="eval",
        )

        self.dataset = self.loader.dataset

        self.scheduler = None

        if torch.cuda.is_available() and self.global_config.gpus != -1:
            self.model.cuda()

        self.optim = AdamW(list(self.model.parameters()), 1e-04)

        torch.manual_seed(self.global_config.seed)

        def train(self):
            for epoch in range(self.train_config.epochs):
                self.epoch = epoch
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
                    )

                    # do loss now
                    loss = output.loss
                    loss.backward()

                    # find score now
                    score = output.question_answering_score
                    logit = torch.nn.functional.softmax(score, dim=-1)
                    logit, pred = logit.max(1)
                    gold = batch["label"]
                    right += (gold.eq(pred.long())).sum()
                    total += float(self.train_config.batch_size)

                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

                    # scheuler
                    if self.scheduler is not None:
                        self.scheduler.step()

                # inference
                if not self.dataset_config.skip_eval:
                    val_acc = inference(self.model, self.inference_loader)

                log_str = (
                    f"\tEpoch {self.epoch}: train {(right/total*100):0.2f} % val"
                    f" {(val_acc):0.2f}%\n"
                )
                log_append(self.log_file, log_str)
                print(log_str)

        def inference(self):
            total = 0.0
            right = 0.0
            preds = torch.Tensor([])
            scores = torch.Tensor([])
            for b in tqdm(self.inference_loader):
                self.model.eval()
                b = self.loader.toCuda(b)
                output = self.model(
                    input_ids=b["input_ids"],
                    visual_feats=b["roi_features"],
                    visual_pos=b["boxes"],
                    attention_mask=b["attention_mask"],
                    token_type_ids=b["token_type_ids"],
                )
                logit = output.question_answering_score
                logit = torch.nn.functional.softmax(logit, dim=-1)
                score = b["label"]
                logit, pred = logit.max(-1)
                right += (score.eq(pred.long())).sum()
                total += float(self.loader.batch_size)
                scores = torch.cat((scores, score.detach().cpu()), dim=0)
                preds = torch.cat((preds, pred.detach().cpu()), dim=0)
            return right / total * 100
