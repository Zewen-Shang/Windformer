import torch
from torch import nn
from torch.backends import cudnn
import numpy as np 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from dataset import *

@torch.no_grad()
def show_predict_img(dataset_test,model):
    model.eval()
    data_item = dataset_test[0]
    input = data_item[0].unsqueeze(0).to(list(model.parameters())[0].device)
    # output,item[1] (predict_steps,1,30,20)
    output = model(input).squeeze(0).cpu()
    diff = torch.abs(output - data_item[1].cpu())
    predict_steps = output.shape[0]
    fig = plt.figure(figsize=(predict_steps,3))
    vmin,vmax = data_item[1].min(),data_item[1].max()
    for i in range(predict_steps):
        ax = fig.add_subplot(3,predict_steps,i+1)
        ax.imshow(np.array(data_item[1][i,0].cpu()),vmin=vmin,vmax=vmax)
        ax = fig.add_subplot(3,predict_steps,6+i+1)
        ax.imshow(np.array(output[i,0]),vmin=vmin,vmax=vmax)
        ax = fig.add_subplot(3,predict_steps,12+i+1)
        ax.imshow(np.array(diff[i,0]))
    plt.savefig("./test.jpg",bbox_inches='tight',pad_inches=0.0,dpi=1200)

model = torch.load("./model/ConvLstm_1.pt")

target_map = 1
input_steps = predict_steps = 6

dataset = get_dataset(target_map,input_steps,predict_steps,debug=False)

for i in range(len(dataset)):
    dataset[i][0] = torch.from_numpy(dataset[i][0]).to(dtype=torch.float)
    dataset[i][1] = torch.from_numpy(dataset[i][1]).to(dtype=torch.float)

cut_pos = int(0.75 * len(dataset))
dataset_train = dataset[:cut_pos]
dataset_test = dataset[cut_pos:]


show_predict_img(dataset_test,model)
