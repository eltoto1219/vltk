# coding=utf-8
import datetime
import os
from collections import namedtuple

import torch
from mmf.common.registry import registry
from modeling_uniter import VisualBertConfig, VisualBertForQuestionAnswering
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AdamW  # LxmertForQuestionAnswering,
from transformers import get_linear_schedule_with_warmup

from dataloader import MMFLoader

torch.manual_seed(1)


def train(
    model, optim, train_data, valid_data, outfile, warmup=None, epochs=8, scheduler=None
):
    add_header = True
    for epoch in range(epochs):
        total = 0.0
        right = 0.0
        for b in tqdm(train_data):
            model.train()
            optim.zero_grad()
            b = train_data.toCuda(b)
            output = model(
                input_ids=b["input_ids"],
                visual_feats=b["roi_features"],
                visual_pos=b["boxes"],
                attention_mask=b["attention_mask"],
                labels=b["label"],
                token_type_ids=b["token_type_ids"],
                image_text_alignment=b["obj_input_ids"],
                extras=b.get("extras", None),
            )
            raise Exception(output.attentions)
            """

            d = cheat({
                'position_embeddings_visual': None,
                'visual_embeddings_type': None,
                'input_ids' : b["input_ids"],
                'visual_embeddings' : b["roi_features"],
                'visual_pos' : b["boxes"],
                'attention_mask' : b["attention_mask"],
                'segment_ids' : b["token_type_ids"],
                'input_mask': None,
                'image_text_alignment': None
            })
            output = model(d)
            """

            labels = b["label"]
            loss = output.loss
            # logit = output["scores"]
            logit = output.question_answering_score
            # loss = torch.nn.functional.cross_entropy(logit, labels)
            loss.backward()

            logit = torch.nn.functional.softmax(logit, dim=-1)
            logit, pred = logit.max(1)
            score = b["label"]
            right += (score.eq(pred.long())).sum()
            total += float(train_data.batch_size)
            # torch.nn..clip_grad_norm_(model.parameters(), 5.0)
            # norm = torch.nn..clip_grad_norm_(model.parameters(), 0.25)
            if warmup is not None:
                warmup.step()
            optim.step()
            for p in optim.param_groups:
                pass
                # print(p["lr"])

            if TEST:
                break

            if scheduler is not None:
                scheduler.step()

        val_acc = inference(model, valid_data)
        if add_header:
            add_header = False
            date = datetime.datetime.now()
            with open(outfile, "a") as f:
                f.write(f"\nName {NAME}\n")
                f.write(f"Date: {date} \n")
                f.flush()

        log_str = (
            f"\tEpoch {epoch}: train {(right/total*100):0.2f} % val"
            f" {(val_acc[0]):0.2f} % roc {(val_acc[1]):0.2f} %\n"
        )
        print(log_str)
        if not TEST:
            with open(outfile, "a") as f:
                f.write(log_str)
                f.flush()

    with open(outfile, "a") as f:
        date = datetime.datetime.now()
        if not TEST:
            f.write(f"Time: {date} \n")
        else:
            f.write(f"(TEST) Time: {date} \n")
        f.flush()

    total = 0.0
    right = 0.0
    preds = torch.Tensor([])
    scores = torch.Tensor([])
    for b in tqdm(loader):
        model.eval()
        b = loader.toCuda(b)
        output = model(
            input_ids=b["input_ids"],
            visual_feats=b["roi_features"],
            visual_pos=b["boxes"],
            attention_mask=b["attention_mask"],
            token_type_ids=b["token_type_ids"],
            image_text_alignment=b["obj_input_ids"],
        )
        """
        d = cheat({
            'position_embeddings_visual': None,
            'visual_embeddings_type': None,
            'input_ids' : b["input_ids"],
            'visual_embeddings' : b["roi_features"],
            'visual_pos' : b["boxes"],
            'attention_mask' : b["attention_mask"],
            'segment_ids' : b["token_type_ids"],
            'input_mask': None,
            'image_text_alignment': None
        })
        output = model(d)
        """

        logit = output.question_answering_score
        # logit = output["scores"]
        logit = torch.nn.functional.softmax(logit, dim=-1)
        score = b["label"]
        logit, pred = logit.max(-1)
        right += (score.eq(pred.long())).sum()
        total += float(loader.batch_size)
        scores = torch.cat((scores, score.detach().cpu()), dim=0)
        preds = torch.cat((preds, pred.detach().cpu()), dim=0)
    return right / total * 100, roc_auc_score(scores.numpy(), preds.numpy())


if __name__ == "__main__":

    outfile = os.path.realpath("p_logs.txt")
    # model = LxmertForQuestionAnswering.from_pretrained(
    #     os.path.realpath("language_heavy"
    # ))
    conf = VisualBertConfig()
    model = VisualBertForQuestionAnswering.from_pretrained(
        os.path.realpath("./hateful_ckp"),
    )

    # model = registry.get_model_class(
    #         "visual_bert"
    #     ).from_pretrained(
    #         "visual_bert.finetuned.hateful_memes"
    #     )
    # model.training_head_type = "finetuning"
    # model.resize_num_qa_labels(2)
    model = model.cuda()
    optim = AdamW(list(model.parameters()), 1e-04)
    # warmup = get_linear_schedule_with_warmup(optim, num_warmup_steps=1000,
    #         num_training_steps=22000)
    warmup = None
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=200, gamma=0.9, last_epoch=-1
    )
    # scheduler = None
    train_data = MMFLoader(
        "train.jsonl",
        "arrow/mmf.arrow",
        objs="objects.txt",
        max_objs=36,
        sent_length=20,
        batch_size=64,
        collate_pt=True,
        num_workers=8,
        percent=1,
    )
    val_data = MMFLoader(
        "dev.jsonl",
        "arrow/mmf.arrow",
        objs="objects.txt",
        max_objs=36,
        sent_length=20,
        batch_size=32,
        collate_pt=True,
        num_workers=0,
    )

    train(
        model,
        optim,
        train_data,
        val_data,
        outfile,
        warmup=warmup,
        epochs=6,
        scheduler=scheduler,
    )
