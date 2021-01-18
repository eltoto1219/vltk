from statistics import mean

from vltk.abc.experiment import Experiment
from vltk.loop import lxmert


class Lxmert(Experiment):

    name: str = "lxmert"
    loops_to_models: dict = {
        lxmert.Lxmert: ["lxmert_qa"],
        lxmert.Lxmert.eval_instance("eval_lxmert"): ["lxmert_qa"],
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
