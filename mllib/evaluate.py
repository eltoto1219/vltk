import os
from dataclasses import dataclass

import torch
# from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm

from .dataloader import BaseLoader


@dataclass
class EvaluateOutputs:
    accuracy: int

    def __init__(self, accuracy):
        self.accuracy = accuracy


class Evaluator:
    def __init__(
        self,
        model_instance,
        dataset_name,
        dataset_config,
        train_config,
        global_config,
        pathes_config,
        loader_config,
        model_config,
        split="eval",
    ):

        self.model = model_instance
        self.train_config = train_config
        self.split = split
        self.log_file = os.path.join(global_config.log_dir, global_config.log_file)
        self.loader = BaseLoader(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            loader_config=loader_config,
            global_config=global_config,
            pathes_config=pathes_config,
            train_config=train_config,
            split=split,
        )

        self.dataset = self.loader.dataset
        self.dry_run = self.train_config.dry_run

    def __call__(self):
        return self.evaluate()

    @torch.no_grad()
    def evaluate(self):
        total = 0.0
        right = 0.0
        preds = torch.Tensor([])
        scores = torch.Tensor([])
        for b in tqdm(self.loader):
            self.model.eval()
            b = self.loader.toCuda(b)
            output = self.model(
                input_ids=b["input_ids"],
                visual_feats=b["roi_features"],
                visual_pos=b["boxes"],
                attention_mask=b["attention_mask"],
                token_type_ids=b["token_type_ids"],
                return_dict=True,
            )
            logit = output.question_answering_score
            logit = torch.nn.functional.softmax(logit, dim=-1)
            score = b["label"]
            logit, pred = logit.max(-1)
            right += (score.eq(pred.long())).sum()
            total += float(self.loader.batch_size)
            scores = torch.cat((scores, score.detach().cpu()), dim=0)
            preds = torch.cat((preds, pred.detach().cpu()), dim=0)

            # check if test run:
            if self.dry_run:
                break

        return EvaluateOutputs(accuracy=(right / total) * 100)
