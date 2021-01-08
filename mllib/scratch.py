# from mllib.compat import Config
# from mllib.configs import Config
# from mllib.loop.lxmert import Lxmert

# c = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
# print(c)

# EvalLxmert = Lxmert.eval_instance("eval_lxmert")
# config = Config()
# test = EvalLxmert(config=config, datasets="gqa", model_dict=None, extra_modules=None)

# print(dir(EvalLxmert))
# print(EvalLxmert.is_train)
# print(EvalLxmert.name)


from mllib.configs import Config
from mllib.maps import dirs, files

l = files.Label()
t = dirs.Textsets()
i = dirs.Imagesets()
e = dirs.Exps()
print(i.avail())
# imageproc = maps.Imageproc()
# lxmert = models.get_model("lxmertforquestionanswering")
# config = Config().data
# add_label_processsor("foo", lambda x: x)
# VQAset.extract(config=config, split="trainval", label_processor="foo")

# print(get_features("foo"))
