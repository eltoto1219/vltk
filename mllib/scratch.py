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


from mllib.decorators import get_duration
from mllib.imageset.frcnn import FRCNNSet


@get_duration
def load_coco_test():
    return FRCNNSet.from_file("/playpen1/home/avmendoz/data/coco2014/frcnn/test.arrow")


load_coco_test()


# lxmert = models.get_model("lxmertforquestionanswering")
# config = Config().data
# add_label_processsor("foo", lambda x: x)
# VQAset.extract(config=config, split="trainval", label_processor="foo")

# print(get_features("foo"))
