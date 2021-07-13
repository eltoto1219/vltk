from itertools import chain

import torch
import vltk.vars as vltk
from vltk.processing import VisnLangProcessor
from vltk.utils.adapters import truncate_and_pad_list


class Span(VisnLangProcessor):
    _keys = (vltk.tokenmap, vltk.span)

    def map_span(self, span, tokenmap, max_len):
        span = list(chain(*map(lambda x: [x[0]] * x[1], zip(span, tokenmap))))
        span = truncate_and_pad_list(span, max_len, 0)
        if self.config.add_visual_cls:
            span = [0] + span[1:]
        return span

    def forward(self, lang_entry, visn_entry, **kwargs):
        img_first = kwargs.get("img_first")
        max_len = self.config.lang.max_visual_seq_length
        tokenmap = visn_entry[vltk.tokenmap]
        span = lang_entry[vltk.span]
        if img_first:
            span = list(map(lambda s: self.map_span(s, tokenmap, max_len), span))
        else:
            span = self.map_span(span, tokenmap, max_len)
        lang_entry[vltk.span] = torch.tensor(span)
        return lang_entry, visn_entry
