from densenet import DenseNet
from resnet import ResNet
from dataset import *

import torch
from torch import nn
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils.data import DataLoader

@torch.no_grad()
def test_model(model,dataloader,device,criterion):
    total_loss = 0.
    total_num = 0
    for imgs,targets in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = model(imgs)
        loss = criterion(output,targets)
        total_loss += loss * len(imgs)
        total_num += len(imgs)
    return total_loss / total_num


def test_time_displace(dataset_test):
    targets = []
    for data_item in dataset_test:
        targets.append(data_item[1])
    targets = torch.tensor([item.detach().numpy() for item in targets])
    output = targets.clone()
    output[3:] = targets[:len(targets)-3]
    return nn.MSELoss()(output,targets).item()

# generate_imgs("power_output")
# generate_imgs("speed")


cudnn.benchmark = True

epoch = 100
criterion = nn.MSELoss()
device = torch.device("cuda")
batch_size = 64
keep_prob = 0.5

dataset,scale = get_dataset_single([16,13])
cut_pos = int(0.5 * len(dataset))
dataset_train = dataset[:cut_pos]
dataset_test = dataset[cut_pos:]


print(test_time_displace(dataset_test))

dataloader_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
dataloader_test = DataLoader(dataset_test,batch_size=batch_size)

#dropout
resNet = DenseNet(30,20,3,20,scale,keep_prob)
resNet.to(device)
optimizer = torch.optim.Adam(resNet.parameters(),lr=1e-3,weight_decay=1e-5)

for i in range(epoch):
    if epoch < 25: lr = 0.1
    if epoch >= 25: lr = 0.01
    if epoch >= 40: lr = 0.001
    if epoch >= 55: lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    resNet.train()
    for imgs,targets in dataloader_train:
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = resNet(imgs)
        loss = criterion(output,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    resNet.eval()
    train_loss,test_loss = 0.,0.
    
    train_loss = test_model(resNet,dataloader_train,device,criterion)
    test_loss = test_model(resNet,dataloader_test,device,criterion)
    print("Epoch %d : Train Loss : %f, Test Loss : %f ." % (i,train_loss,test_loss))

# y = denseNet(x)

# print(y)

