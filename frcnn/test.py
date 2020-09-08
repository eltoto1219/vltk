import torch

from frcnn import GeneralizedRCNN as Model


hi = Model.from_pretrained_v2("./frcnn-vg-finetuned")
torch.save(hi.state_dict(), "pytorch_model.bin")
