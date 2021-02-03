import math

import torch
import torch.nn as nn
from vltk import metrics
from vltk.abc.simple import SimpleExperiment
from vltk.modeling import (DistilledVisionTransformer,
                           LxmertForQuestionAnswering)

__all__ = ["DeitConv"]


class Conv(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.proj = nn.Conv2d(
            kwargs["in_chans"],
            kwargs["out_chans"],
            kernel_size=kwargs["patch_size"],
            stride=kwargs["patch_size"],
        )
        n_patches = math.floor(kwargs["embed_dim"] / kwargs["patch_size"])
        foo = math.floor(n_patches ** 2 / kwargs["patch_size"])
        self.proj_2 = nn.Linear(n_patches * foo, kwargs["embed_dim"])
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = self.act(x)
        x = self.proj_2(x)
        return x


class DeitConv(SimpleExperiment):
    """
    iter_outputs contain combined batch dictionary and ouput dictionary from forward
    """

    name = "deitconv"
    model_list = [
        ("lxmert", LxmertForQuestionAnswering),
        ("deit", DistilledVisionTransformer),
        ("conv", Conv),
    ]

    def forward(self, batch) -> dict:

        self.toCuda(batch, device=self.deit_dev)

        output = None
        images = torch.split(batch["image"], 1, dim=0)
        for x in images:
            temp_output = self.deit(x)
            if output is None:
                output = temp_output
            else:
                for k, v in output.items():
                    output[k] = torch.cat((v, temp_output[k]), dim=0)

        attn_outputs = {}
        for a, img_id in zip(output["attns"], batch["img_id"]):
            attn_outputs[img_id] = a.detach().cpu().tolist()
        dist_token = output["dist"]
        feats = output["feats"]
        feats = self.conv(feats.unsqueeze(1))
        mimic_feats = torch.cat((dist_token.unsqueeze(1), feats), dim=1)
        boxes = torch.zeros(mimic_feats.shape[0], mimic_feats.shape[1], 4)

        batch["roi_features"] = mimic_feats
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
            "attentions": attn_outputs,
        }

    def iter_tqdm(self, forward_outputs, train=True):
        bz = len(forward_outputs["input_ids"])
        acc = metrics.accuracy(forward_outputs["score"], forward_outputs["label"])
        lr = [f"{l:.3e}" for l in self.get_lr()]
        loss = f"{forward_outputs['losses']:.3f}"
        return {"loss": loss, "acc": acc, "bz": bz, "lr": lr}

    def iter_save(forward_outputs, train=True):
        return {"attentions": forward_outputs["attentions"]}

    def epoch_logstr(self, loop_outputs, train=True):
        return {
            "acc": loop_outputs["acc"],
            "bz": loop_outputs["bz"],
            "loss": f"{loop_outputs['losses']}.3f",
        }
