from mllib.abc.experiment import Experiment
from mllib.loop import lxmert


class Lxmert(Experiment):

    name: str = "runlxmert"
    loops_to_models: dict = {lxmert.Lxmert: ["lxmert"], lxmert.Lxmert.eval_instance("eval_lxmert"): ["lxmert"]}

    def loginfo(self, **kwargs):
        '''
        kwargs will be  loop output from every run, with the run name being the key
        '''
        logstr = ''
        for k, v in kwargs.items():
            logstr += f'{k}: accuracy={v.accuracy} '
        return logstr
