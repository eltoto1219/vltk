# from vltk.compat import Config
# from vltk.configs import Config
# from vltk.loop.lxmert import Lxmert

# c = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
# print(c)

# EvalLxmert = Lxmert.eval_instance("eval_lxmert")
# config = Config()
# test = EvalLxmert(config=config, datasets="gqa", model_dict=None, extra_modules=None)

# print(dir(EvalLxmert))
# print(EvalLxmert.is_train)
# print(EvalLxmert.name)


# from transformers import LxmertModel

# import vltk.loop.data
# from modeling.configs import Get
# from vltk.decorators import get_duration
# from vltk.visndatasetadapter.frcnn import FRCNNSet

# lxmert = models.get_model("lxmertforquestionanswering")
# config = Config().data
# add_label_processsor("foo", lambda x: x)
# VQAset.extract(config=config, split="trainval", label_processor="foo")

# print(get_features("foo"))
import vltk

vltk.imgid
