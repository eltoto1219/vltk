from itertools import chain

import torch
import vltk
from vltk.processing import VisnProcessor
from vltk.utils.adapters import rescale_box, truncate_and_pad_list


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

        if not self.from_transformers:
            self.disable_padding()
            text = list(
                map(
                    lambda x: x.ids,
                    self.tokenizer.encode_batch(
                        entry.pop(vltk.text), add_special_tokens=False
                    ),
                )
            )
            self.enable_padding()
        else:
            text = self.tokenizer(
                entry.pop(vltk.text),
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

        tokenmap = list(map(lambda x: len(x), text))
        tokenmap = torch.tensor(truncate_and_pad_list(tokenmap, max_len, 0))
        entry[vltk.tokenmap] = tokenmap
        if not self.from_transformers:
            pad_id = self.tokenizer.token_to_id(self.tokenizer.pad_token)
            text = truncate_and_pad_list(list(chain(*text)), max_len - 1, pad_id)
            text += [self.tokenizer.token_to_id(self.tokenizer.sep_token)]
        else:
            pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            text = truncate_and_pad_list(list(chain(*text)), max_len - 1, pad_id)
            text += [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]

        entry[vltk.text] = torch.tensor(text)
        return entry


class OCRBox(VisnProcessor):
    _keys = (vltk.tokenbox, vltk.tokenmap)

    def forward(self, entry, **kwargs):
        max_len = self.config.lang.max_visual_seq_length
        tokenboxes = entry.pop(vltk.tokenbox)
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
        return entry
