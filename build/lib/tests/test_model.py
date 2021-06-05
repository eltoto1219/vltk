import torch
from vltk.modeling.botnet_v1 import BottleStack
from vltk.modeling.botnet_v2 import ResNet50
from vltk.modeling.botnet_v3 import MyBotNet
from vltk.utils import get_nvidia_gpu_memory

scaler = torch.cuda.amp.GradScaler()

size = 768
bz = 4

print("init model")
# model1 = BottleStack(dim=3, fmap_size=size)
model2 = MyBotNet(resolution=(size, size))
model2.to(torch.device(3))
# print("Bottle Stack")
# for n, p in model1.named_parameters():
#     print(n, p.shape)
# print("resent")
# for n, p in model2.named_parameters():
#     print(n, p.shape)


# print("data")
x = torch.rand(bz, 3, size, size)
x = x.to(torch.device(3))

with torch.cuda.amp.autocast():
    output = model2(x)
    print(output.shape)
