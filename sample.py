from mllib import metrics
from mllib.abc.experiment import Experiment
from mllib.abc.loop import Loop
from mllib.outputs import LoopOutputs


class SampleLoop(Loop):
    name: str = "sample"
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
                "bz": flatten_bz
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
                    "sizes"
                ]
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


class SampleExperiment(Experiment):

    # the name will be registered for use on the command line. IE: run exp sample gqa ....
    name: str = "sample"
    loops_to_models: dict = {
        # we train lxmert. why in brackets?, you can actually train arbitrary models at once!
        SampleLoop: ["lxmert"],
        # SampleLoop will train lxmert, but what about eval?
        # luckily, we have an eval factory in SampleLoop
        # the only overhead is the name "eval_sample" which we must define for the new class!
        SampleLoop.eval_instance("eval_sample"): ["lxmert"]
    }

    # extra_modules = {"myextra": torch.nn.Linear(1, 2)}
    # "myextra" will be defined as an attribute in Sample(Loop) which will point to the nn.Module

    def loginfo(self, **kwargs):
        '''
        kwargs will be  loop output from every run, with the run name being the key
        '''
        logstr = ''
        for k, v in kwargs.items():
            logstr += f'{k}: accuracy={v.accuracy} '
        return logstr
