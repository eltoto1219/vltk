from vltk import metrics
from vltk.abc.loop import Loop
from vltk.outputs import LoopOutputs


class Lxmert(Loop):
    name: str = "lxmert"
    is_train: bool = True

    def loop(self, batch, model_outputs):
        acc = metrics.accuracy(model_outputs.question_answering_score, batch["labels"])
        loop_outputs = LoopOutputs(accuracy=acc, losses=model_outputs.loss)
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
        self.toCuda(batch, device=self.config.gpu)
        batch["return_dict"] = True
        model_outputs = self.lxmert(
            input_ids=batch["input_ids"],
            visual_feats=batch["roi_features"],
            visual_pos=batch["boxes"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            return_dict=batch["return_dict"],
            labels=batch["labels"],
        )
        return model_outputs
