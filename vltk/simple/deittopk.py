import torch
from vltk import metrics
from vltk.abc.simple import SimpleExperiment
from vltk.modeling import (DistilledVisionTransformer,
                           LxmertForQuestionAnswering)

__all__ = ["DeitTopk"]


class DeitTopk(SimpleExperiment):
    """
    iter_outputs contain combined batch dictionary and ouput dictionary from forward
    """

    name = "deittopk"
    model_list = [
        ("lxmert", LxmertForQuestionAnswering),
        ("deit", DistilledVisionTransformer),
    ]

    def forward(self, batch) -> dict:

        self.toCuda(batch, device=self.config.aux_gpu)
        output = self.deit(batch["image"])

        attn_outputs = {}
        for a, img_id in zip(output["attns"], batch["img_id"]):
            attn_outputs[img_id] = a.detach().cpu().tolist()
        dist_token = output["dist"]
        feats = output["feats"]
        feats = feats.to(torch.device(self.config.gpu))
        feats_topk = torch.topk(
            feats, self.config.models.lxmert.topk_patches, dim=1, sorted=False
        )
        feat_idxs = feats_topk.indices
        # we want to preserve order, so  here we sort the indicies, an then select
        feats = feats_topk.values
        topk = {img_id: topk_i for img_id, topk_i in zip(batch["img_id"], feat_idxs)}
        for a, img_id in zip(output["attns"], batch["img_id"]):
            attn_outputs[img_id] = a.detach().cpu().tolist()
        dist_token = output["dist"]
        dist_token = dist_token.to(torch.device(self.config.gpu))
        mimic_feats = torch.cat((dist_token.unsqueeze(1), feats), dim=1)
        boxes = torch.zeros(mimic_feats.shape[0], mimic_feats.shape[1], 4).to(
            torch.device(self.config.gpu)
        )

        batch["roi_features"] = mimic_feats
        batch["boxes"] = boxes

        self.transpose_img2txt(batch, img_keys=["boxes", "roi_features"])

        self.toCuda(batch, device=self.config.gpu)
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
            "topk_inds": topk,
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
