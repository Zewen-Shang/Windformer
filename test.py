import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt



coords_d = torch.tensor([0,1,2])
coords_h = torch.tensor([0,1,2,3])
coords_w = torch.tensor([0,1,2,3,4])

a = torch.meshgrid([coords_d,coords_h,coords_w])



class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.para = torch.rand((4,1))
        self.para = nn.Parameter(self.para)
        self.buff = torch.tensor([[-1]])

    def forward(self,x):
        print("para",self.para)
        print("buff",self.buff)
        return x @ self.para + self.buff



net = Net()

optim = torch.optim.SGD(net.parameters(),lr=0.1)
schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optim,20)

lr = []

for i in range(100):
    lr.append(schedular.get_lr()[0])
    schedular.step()

plt.plot(lr)
plt.savefig("./test.jpg")


x = torch.nn.Parameter(x)



input_steps = 4
shift_size = 2

h_slices = (slice(0, -input_steps),
            slice(-input_steps, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, input_steps),
            slice(-input_steps, -shift_size),
            slice(-shift_size, None))


for h in h_slices:
    for w in w_slices:
        print()

coords_h = torch.arange(7)
coords_w = torch.arange(7)

coords = torch.meshgrid([coords_h, coords_w])
coords = torch.stack(coords)

print()
