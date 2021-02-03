import math

import torch
from vltk import metrics
from vltk.abc.simple import SimpleExperiment
from vltk.modeling import (DistilledVisionTransformer,
                           LxmertForQuestionAnswering)

__all__ = ["DeitLxmertSimple"]


class DeitLxmertSimple(SimpleExperiment):
    """
    iter_outputs contain combined batch dictionary and ouput dictionary from forward
    """

    name = "deitlxmert"
    model_list = [
        ("lxmert", LxmertForQuestionAnswering),
        ("deit", DistilledVisionTransformer),
        ("connector", torch.nn.Linear(11520, 2048)),
        ("dist_proj", torch.nn.Linear(576, 2048)),
    ]

    def forward(self, batch) -> dict:

        self.toCuda(batch, device=self.deit_dev)

        output = self.deit(batch["image"])

        attn_outputs = {}
        for a, img_id in zip(output["attns"], batch["img_id"]):
            attn_outputs[img_id] = a.detach().cpu().tolist()
        dist_token = output["dist"]
        feats = output["feats"]
        feats = feats.flatten(start_dim=-2, end_dim=-1)
        feats = feats.view(
            feats.shape[0],
            math.floor(
                self.config.models.deit.embed_dim / self.config.models.deit.patch_size
            ),
            -1,
        )
        dist_proj = self.dist_proj(dist_token).unsqueeze(1)
        feats = self.connector(feats)
        mimic_feats = torch.cat((dist_proj, feats), dim=1)
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
