import datetime
import os

import jax
import jax.numpy as jnp
import torch
# from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AdamW  # LxmertForQuestionAnswering,

from mllib.models import CONFIGS, KNOWN_MODELS, load_pretrained

from .dataloader import BaseLoader
from .evaluate import Evaluator

MODEL_SIZES = {
    "ViT-B_16": 86_567_656,
    "ViT-B_32": 88_224_232,
    "ViT-L_16": 304_326_632,
    "ViT-L_32": 306_535_400,
    "ViT-H_14": 632_045_800,
    "testing": 2985,
}


def run_vit(vit, batch):
    # return vit_outputs
    pass


def init_vit(model_config, train_config, global_config, pathes_config):
    vit_variant = model_config.vit_variant
    vit_pretrained_dir = os.path.join(
        global_config.data_dir, pathes_config.vit_pretrained_dir
    )

    # return vit
    rng = jax.random.PRNGKey(0)
    VisualTransformer = KNOWN_MODELS[vit_variant].partial(num_classes=0)
    output, initial_params = VisualTransformer.init_by_shape(
        rng, [((2, 832, 832, 3), jnp.float32)]
    )
    # pretrained_path = os.path.join(args.vit_pretrained_dir, f"{args.model}.npz")
    # params = load_pretrained(
    #     pretrained_path=pretrained_path,
    #     init_params=params,
    #     model_config=CONFIGS[vit_variant],
    #     logger=logger,
    # )

    print(type(VisualTransformer))

    raise Exception(output, initial_params["Transformer"].keys())


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

        if self.dataset_config.use_raw_imgs:
            self.vit = init_vit(model_config, train_config, global_config, pathes_config)

        self.loader = BaseLoader(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            loader_config=loader_config,
            global_config=global_config,
            pathes_config=pathes_config,
            train_config=train_config,
            split=dataset_config.split,
        )

        self.split = dataset_config.split
        self.eval_split = dataset_config.eval_split
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

            if self.dataset_config.use_raw_imgs:
                vit_output = run_vit(batch)
                raise Exception(type(vit_output))
                batch["features"] = vit_output
            else:
                batch["features"] = batch["roi_features"]

            output = self.model(
                input_ids=batch["input_ids"],
                visual_feats=batch["features"],
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
