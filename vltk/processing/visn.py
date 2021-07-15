from itertools import chain

import torch
import vltk.vars as vltk
from vltk.processing import VisnProcessor
from vltk.utils.adapters import (imagepoints_to_mask, rescale_box,
                                 resize_binary_mask, seg_to_mask,
                                 truncate_and_pad_list)


class PolygonProcessor(VisnProcessor):
    def forward(self, entry, **kwargs):
        polykey = vltk.polygons
        size = entry[vltk.size]
        if vltk.rawsize not in entry:
            rawsize = size
        else:
            rawsize = entry[vltk.rawsize]
        segs = torch.stack(
            list(
                map(
                    lambda x: resize_binary_mask(seg_to_mask(x, *rawsize), size),
                    entry.pop(polykey),
                ),
            )
        )

        segs = segs[: min(len(segs), self.config.lang.max_visual_seq_length)]
        segs = torch.nn.functional.pad(
            segs,
            (0, 0, 0, 0, 0, self.config.lang.max_visual_seq_length - len(segs)),
        )
        entry[vltk.segmentation] = segs
        return entry


class RLEProcessor(VisnProcessor):
    def forward(self, entry, **kwargs):
        rlekey = vltk.RLE
        segs = torch.stack(
            list(
                map(
                    lambda x: resize_binary_mask(
                        imagepoints_to_mask(x, entry[vltk.rawsize]),
                        torch.as_tensor(entry[vltk.size]),
                    ),
                    entry.pop(rlekey),
                )
            )
        )
        segs = segs[: min(len(segs), self.config.lang.max_visual_seq_length)]
        segs = torch.nn.functional.pad(
            segs,
            (0, 0, 0, 0, 0, self.config.lang.max_visual_seq_length - len(segs)),
        )
        entry[vltk.segmentation] = segs


class AuxTokenize(VisnProcessor):
    _keys = vltk.text

    def enable_padding(self):
        self.tokenizer.enable_padding(
            length=self.config.lang.max_seq_length,
            direction=self.config.lang.pad_direction,
            pad_id=self.tokenizer.token_to_id(self.tokenizer.pad_token),
        )

    def disable_padding(self):
        self.tokenizer.no_padding()

    def forward(self, entry, **kwargs):
        max_len = self.config.lang.max_visual_seq_length
        text = entry.pop(vltk.text)
        if self.config.add_visual_cls:
            text = [self.tokenizer.cls_token] + text

        if not self.from_transformers:
            self.disable_padding()
            unk_id = self.tokenizer.token_to_id(self.tokenizer.unk_token)
            text = list(
                map(
                    lambda x: x.ids,
                    self.tokenizer.encode_batch(text, add_special_tokens=False),
                )
            )
            self.enable_padding()
        else:
            unk_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
            text = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

        text = list(map(lambda x: x if x else [unk_id], text))

        tokenmap = list(map(lambda x: len(x), text))
        if len(tokenmap) >= max_len:
            tokenmap = tokenmap[: max_len - 1]

        assert 0 not in tokenmap
        tokenmap = torch.tensor(
            truncate_and_pad_list(tokenmap, max_len, self.config.lang.ignore_id)
        )
        entry[vltk.tokenmap] = tokenmap
        text = list(chain(*text))
        visual_attention_mask = torch.tensor(
            [1] * min(max_len, len(text)) + [0] * max(0, max_len - len(text))
        )
        entry["visual_attention_mask"] = visual_attention_mask
        if not self.from_transformers:
            pad_id = self.tokenizer.token_to_id(self.tokenizer.pad_token)
            text = truncate_and_pad_list(text, max_len - 1, pad_id)
            text += [self.tokenizer.token_to_id(self.tokenizer.sep_token)]
        else:
            pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            text = truncate_and_pad_list(text, max_len - 1, pad_id)
            text += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]

        entry[vltk.text] = torch.tensor(text)
        return entry


class OCRBox(VisnProcessor):
    _keys = (vltk.tokenbox, vltk.tokenmap)

    def forward(self, entry, **kwargs):
        max_len = self.config.lang.max_visual_seq_length
        tokenboxes = entry.pop(vltk.tokenbox)
        if self.config.add_visual_cls:
            tokenboxes = [[0, 0, *entry[vltk.rawsize]]] + tokenboxes
        tokenmap = entry.get(vltk.tokenmap)
        tokenboxes = list(
            chain(
                *map(
                    lambda x: [x[0]] * x[1],
                    zip(tokenboxes, tokenmap),
                )
            )
        )
        tokenboxes = truncate_and_pad_list(tokenboxes, max_len, [0, 0, 0, 0])
        tokenboxes = torch.tensor(tokenboxes)
        if vltk.size in entry:
            tokenboxes = rescale_box(tokenboxes, entry[vltk.scale])
        entry[vltk.tokenbox] = tokenboxes
        return entry


class TokenLabels(VisnProcessor):
    _keys = (vltk.label, vltk.tokenmap)

    def forward(self, entry, **kwargs):
        max_len = self.config.lang.max_visual_seq_length
        labels = entry.get(vltk.label)
        if self.config.add_visual_cls:
            labels = [""] + labels
        tokenmap = entry.get(vltk.tokenmap)
        labels = list(
            chain(
                *map(
                    lambda x: [x[0]] * x[1],
                    zip(labels, tokenmap),
                )
            )
        )
        if len(labels) >= max_len:
            labels = labels[: max_len - 1]
        entry[vltk.label] = labels
        return entry


class OCRBoxFixed(VisnProcessor):
    _keys = (vltk.tokenbox, vltk.tokenmap, vltk.rawsize)

    def forward(self, entry, **kwargs):
        max_len = self.config.lang.max_visual_seq_length
        tokenboxes = entry.pop(vltk.tokenbox)
        raw_w, raw_h = entry[vltk.rawsize]
        scale = (1000 / raw_w, 1000 / raw_h)
        if self.config.add_visual_cls:

            tokenboxes = [[0, 0, raw_w, raw_h]] + tokenboxes
        tokenmap = entry.get(vltk.tokenmap)
        tokenboxes = list(
            chain(
                *map(
                    lambda x: [x[0]] * x[1],
                    zip(tokenboxes, tokenmap),
                )
            )
        )
        tokenboxes = truncate_and_pad_list(tokenboxes, max_len, [0, 0, 0, 0])
        tokenboxes = torch.tensor(tokenboxes)
        tokenboxes = torch.clamp(rescale_box(tokenboxes, scale), min=0, max=1000)
        entry[vltk.tokenbox] = tokenboxes
        return entry


class XYWHtoXYXY(VisnProcessor):
    def forward(self, entry, **kwargs):
        for k in (vltk.tokenbox, vltk.box):
            if k in entry:
                box = entry[k]
                try:
                    box[:, -2:] += box[:, :2]
                except Exception:
                    raise Exception(box)
                entry[k] = box
        return entry
