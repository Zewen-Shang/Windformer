from resnet import ResNet, ResNet_Single
from dataset import *
from utils import MAPE

import torch
from torch import nn
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau,ExponentialLR
from torch.utils.tensorboard import SummaryWriter   

@torch.no_grad()
def test_model(model,dataloader,device):
    total_MSE,total_MAPE = 0.,0.
    total_num = 0
    for imgs,targets in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = model(imgs)
        mse = nn.MSELoss()(output,targets)
        mape = MAPE(targets,output)
        total_MSE += mse * len(imgs)
        total_MAPE += mape * len(imgs)
        total_num += len(imgs)
    return total_MSE / total_num,total_MAPE / total_num


def test_time_displace(dataset_test):
    targets = []
    for data_item in dataset_test:
        targets.append(data_item[1])
    targets = torch.tensor([item.detach().numpy() for item in targets])
    output = targets.clone()
    output[3:] = targets[:len(targets)-3]
    for i in range(3):
        output[i] = output[0]
    return nn.MSELoss()(output,targets).item()

# generate_imgs("power_output")
# generate_imgs("speed")


generate_imgs("speed")

cudnn.benchmark = True

epoch = 5
criterion = nn.MSELoss()
device = torch.device("cuda")
batch_size = 64
drop_prob = 0.1

dataset = get_dataset()
cut_pos = int(0.5 * len(dataset))
dataset_train = dataset[:cut_pos]
dataset_test = dataset[cut_pos:]

# print(test_time_displace(dataset_test))

dataloader_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=10)
dataloader_test = DataLoader(dataset_test,batch_size=batch_size,num_workers=10)

#dropout
resNet = ResNet(depth=2,drop_prob=drop_prob)
resNet.to(device)

lr = 1e-4
optimizer = torch.optim.Adam(resNet.parameters(),lr=lr,weight_decay=1e-3)

writer = SummaryWriter()

for i in range(epoch):
    if(i == 20):
        lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # if(i == 70):
    #     lr *= 0.1
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr

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

    train_MSE,train_MAPE = test_model(resNet,dataloader_train,device)
    test_MSE,test_MAPE = test_model(resNet,dataloader_test,device)
    print("Epoch %d : Train MSE : %f, Train MAPE : %f , Test MSE : %f , Test MAPE : %f , Lr : %f . " % (i,train_MSE,train_MAPE,test_MSE, test_MAPE ,optimizer.param_groups[0]['lr']))
    writer.add_scalar('train_MSE', train_MSE.item(), i)
    writer.add_scalar('train_MAPE', train_MAPE.item(), i)
    writer.add_scalar('test_MSE', test_MSE.item(), i)
    writer.add_scalar('test_MAPE', test_MAPE.item(), i)


resNet.eval()

with torch.no_grad():
    fig = torch.zeros((30,20),device=device)
    for imgs,targets in dataloader_test:
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = resNet(imgs)
        output = ((output - targets) ** 2).sum(dim=0)
        fig += output

    plt.imshow(fig.to("cpu").detach().numpy())

    plt.colorbar(label='max value test')
    plt.savefig("./test.jpg")

print()