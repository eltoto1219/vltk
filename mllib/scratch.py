

from mllib.compat import Config
from mllib.configs import GlobalConfig
from mllib.loop.lxmert import Lxmert

c = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
print(c)

EvalLxmert = Lxmert.eval_instance("eval_lxmert")
config = GlobalConfig()
test = EvalLxmert(config=config, datasets="gqa", model_dict=None, extra_modules=None)

print(dir(EvalLxmert))
print(EvalLxmert.is_train)
print(EvalLxmert.name)
