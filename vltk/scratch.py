if __name__ == "__main__":
    import inspect
    import os
    import sys

    import tokenizers
    import transformers
    from transformers import RobertaTokenizerFast

    VOCABPATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "libdata/bert-base-uncased-vocab.txt")
    ).replace("loader/", "")

    TOKENIZERS = {
        m[0]: m[1]
        for m in inspect.getmembers(sys.modules["tokenizers"], inspect.isclass)
        if "Tokenizer" in m[0]
    }
    # FAST = {
    #     m[0]: m[1]
    #     for m in inspect.getmembers(sys.modules["transformers"], inspect.isclass)
    #     if "FastTokenizer" in m[0]
    # }

    bwpt = TOKENIZERS["ByteLevelBPETokenizer"](VOCABPATH)
    # uspt = TOKENIZERS["SentencePieceUnigramTokenizer"]()
    # bwpt.add_special_tokens(["[UNK]"])

    sent = "Hello, World! my name is Antonio Mendoza. I was born in 01/12/2000. Don't underestimate VLTK"
    # print(sent)
    # encoding = bwpt.encode(sent)  # _batch(sent)
    encoding = bwpt.encode(sent)
    print(encoding)

    # print(dir(bwpt))
    # print("ids")
    # # print(encoding.ids)
    # print([e.ids for e in encoding])
    # print("offsets")
    # print(encoding.offsets)
    # print("tokens")
    # print([bwpt.id_to_token(w) for w in encoding.ids])
    """
        if is_visnlang:
            # TODO this is slow right now, broken somehow.
            # I will need to fix
            filtered_self = self.filter(lambda x: x[vltk.imgid] in imgids)
            new_map = defaultdict(list)
            for i, x in enumerate(filtered_self):
                new_map[x[vltk.imgid]].append(i)
    """
