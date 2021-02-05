import math

import torch
from vltk import metrics
from vltk.abc.simple import SimpleExperiment
from vltk.modeling import LxmertForQuestionAnswering, MyBotNet

__all__ = ["DeitLxmertSimple"]


class DeitLxmertSimple(SimpleExperiment):
    """
    iter_outputs contain combined batch dictionary and ouput dictionary from forward
    """

    name = "botnetlxmert"
    model_list = [
        ("lxmert", LxmertForQuestionAnswering),
        ("botnet", MyBotNet),
    ]

    def forward(self, batch) -> dict:

        self.toCuda(batch, device=self.botnet_dev)

        output = self.botnet(batch["image"])

        boxes = torch.zeros(output.shape[0], output.shape[1], 4)

        batch["roi_features"] = output
        batch["boxes"] = boxes

        self.transpose_img2txt(batch, img_keys=["boxes", "roi_features"])

        self.toCuda(batch, device=self.lxmert_dev)
        model_outputs = self.lxmert(
            input_ids=batch["input_ids"],
            visual_feats=batch["roi_features"],
            visual_pos=batch["boxes"],
            attention_mask=batch["text_attention_mask"],
            token_type_ids=batch["type_ids"],
            return_dict=True,
            labels=batch["label"],
        )

        return {
            "losses": model_outputs.loss,
            "score": model_outputs.question_answering_score,
            "input_ids": batch["input_ids"],
            "label": batch["label"],
        }

    def iter_tqdm(self, forward_outputs, train=True):
        bz = len(forward_outputs["input_ids"])
        acc = metrics.accuracy(forward_outputs["score"], forward_outputs["label"])
        lr = [f"{l:.3e}" for l in self.get_lr()]
        loss = f"{forward_outputs['losses']:.3f}"
        return {"loss": loss, "acc": acc, "bz": bz, "lr": lr}

    def iter_save(forward_outputs, train=True):
        return {}

    def epoch_logstr(self, loop_outputs, train=True):
        return {
            "acc": loop_outputs["acc"],
            "bz": loop_outputs["bz"],
            "loss": f"{loop_outputs['losses']}.3f",
        }
