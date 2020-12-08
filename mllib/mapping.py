from .factories import frcnn_factory, lxmert_factory
from .processors import (img_to_tensor, load_temp_gqa, load_temp_lxmert,
                         process_answer_default)

__all__ = ["NAME2PROCESSOR", "NAME2MODEL"]

NAME2PROCESSOR = {"raw_img": img_to_tensor, "default_ans": process_answer_default}
NAME2MODEL = {"lxmert": lxmert_factory, "frcnn": frcnn_factory}
