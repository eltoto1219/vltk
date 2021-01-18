import torch
from vltk import metrics
from vltk.abc.loop import Loop


class Lxmert(Loop):
    name: str = "lxmert"
    is_train: bool = True

    def loop(self, batch, model_outputs):
        acc = metrics.accuracy(model_outputs.question_answering_score, batch["label"])
        loop_outputs = {"accuracy": acc, "losses": model_outputs.loss}
        flatten_bz = len(batch["input_ids"])
        self.tqdm_update(
            {
                "acc": f"{acc:.3f}%",
                "lrs": [f"{l:.3e}" for l in self.get_lr()],
                "loss": f"{model_outputs.loss:.3f}",
                "bz": flatten_bz,
            }
        )

        return loop_outputs

    def forward(self, batch):
        if self.config.data.img_first:
            self.dataset.transpose_img2txt(
                batch,
                img_keys=[
                    "roi_features",
                    "boxes",
                    "attr_ids",
                    "attr_probs",
                    "obj_ids",
                    "obj_probs",
                    "preds_per_image",
                    "sizes",
                ],
            )

        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["type_ids"] = torch.stack(batch["type_ids"])
        batch["text_attention_mask"] = torch.stack(batch["text_attention_mask"])
        batch["label"] = torch.stack(batch["label"])
        self.toCuda(batch, device=self.config.gpu)

        model_outputs = self.lxmert_qa(
            input_ids=batch["input_ids"],
            visual_feats=batch["roi_features"],
            visual_pos=batch["boxes"],
            attention_mask=batch["text_attention_mask"],
            token_type_ids=batch["type_ids"],
            return_dict=True,
            labels=batch["label"],
        )
        return model_outputs
