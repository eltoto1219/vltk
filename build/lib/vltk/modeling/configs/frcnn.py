from vltk.compat import Config

__all__ = ["FRCNNConfig"]
FRCNNConfig = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

if __name__ == "__main__":
    print(FRCNNConfig)
