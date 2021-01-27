import random

from vltk.abc.simple import SimpleExperiment
from vltk.modeling import LxmertModel

__all__ = ["LxmertSimple"]


class LxmertSimple(SimpleExperiment):
    i = 0
    """
    iter_outputs contain combined batch dictionary and ouput dictionary from forward
    """

    name = "lxmert_simple"
    model_list = [("lxmert", LxmertModel)]

    def forward(self, batch) -> dict:
        assert hasattr(self, "lxmert")
        self.i += random.randint(0, 100)
        i = self.i

        acc = metrics.accuracy(model_outputs.question_answering_score, batch["label"])
        return {"i": i}

    def iter_tqdm(self, iter_outputs, train=True):
        return iter_outputs

    def iter_save(iter_outputs, train=True):
        return iter_outputs

    def epoch_logstr(self, iter_outputs, train=True):
        return iter_outputs
