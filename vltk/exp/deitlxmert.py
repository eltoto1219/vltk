from statistics import mean

import torch
import torch.nn.functional as F
from vltk import metrics
from vltk.abc.experiment import Experiment
from vltk.abc.loop import Loop


class DeitLxmertLoop(Loop):
    name: str = "deitlxmert"
    is_train: bool = True

    def loop(self, batch, model_outputs):
        acc = metrics.accuracy(model_outputs.question_answering_score, batch["label"])
        flatten_bz = len(batch["input_ids"])
        loop_outputs = {"accuracy": acc, "losses": model_outputs.loss, "bz": flatten_bz}
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

        self.toCuda(batch, device=self.config.aux_gpu)

        output = self.deit(batch["image"])

        # attns = output["attns"]
        dist_token = output["dist"]
        feats = output["feats"]
        feats = feats.flatten(start_dim=-2, end_dim=-1)
        feats = feats.view(feats.shape[0], self.config.models.deit.patch_size, -1)
        feats = feats.to(torch.device(self.config.gpu))
        dist_token = dist_token.to(torch.device(self.config.gpu))
        dist_proj = self.dist_proj(dist_token).unsqueeze(1)
        feats = self.connector(feats)
        mimic_feats = torch.cat((dist_proj, feats), dim=1)
        boxes = torch.zeros(mimic_feats.shape[0], mimic_feats.shape[1], 4).to(
            torch.device(self.config.gpu)
        )

        batch["roi_features"] = mimic_feats
        batch["boxes"] = boxes

        self.dataset.transpose_img2txt(batch, img_keys=["boxes", "roi_features"])

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


class DeitLxmertExp(Experiment):
    name: str = "deitlxmert"
    loops_to_models: dict = {
        DeitLxmertLoop: ["lxmert_qa", "deit"],
        DeitLxmertLoop.eval_instance("eval_deittlxmert"): ["lxmert_qa", "deit"],
    }
    extra_modules = {
        "connector": torch.nn.Linear(16128, 2048),
        "dist_proj": torch.nn.Linear(576, 2048),
    }

    def loginfo(self, info_dict):
        """
        info_dict will be  loop output from every run, with the run name being the key
        each value from the loop output will be a list of the value collected from the
        loopoutput from every batch

        """
        logstr = ""
        for k, v in info_dict.items():
            logstr += f"{k}={mean(map(lambda x: float(x), v))} "
        return logstr
