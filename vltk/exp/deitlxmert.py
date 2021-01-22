from statistics import mean

import torch
import torch.nn.functional as F
from vltk import metrics
from vltk.abc.experiment import Experiment
from vltk.abc.loop import Loop


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(
        self,
        base_criterion: torch.nn.Module,
        teacher_model: torch.nn.Module,
        distillation_type: str,
        alpha: float,
        tau: float,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == "none":
            return base_loss

        if outputs_kd is None:
            raise ValueError(
                "When knowledge distillation is enabled, the model is "
                "expected to return a Tuple[Tensor, Tensor] with the output of the "
                "class_token and the dist_token"
            )
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == "soft":
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = (
                F.kl_div(
                    F.log_softmax(outputs_kd / T, dim=1),
                    F.log_softmax(teacher_outputs / T, dim=1),
                    reduction="sum",
                    log_target=True,
                )
                * (T * T)
                / outputs_kd.numel()
            )
        elif self.distillation_type == "hard":
            distillation_loss = F.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1)
            )

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class DeitLxmertLoop(Loop):
    name: str = "deitlxmert"
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

        self.toCuda(batch, device=self.aux_model_device)

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

        self.toCuda(batch, device=self.main_device)
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
