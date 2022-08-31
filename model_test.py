from distutils.command.config import config
import torch
from torch import nn
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
from dataset import *
from utils import MAPE
from config import args,models

@torch.no_grad()
def test_model(model,dataloader,device):
    total_MSE,total_MAPE = 0.,0.
    total_num = 0
    for imgs,targets in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device).squeeze()
        output = model(imgs).squeeze()
        mse = nn.MSELoss()(output,targets)
        mape = MAPE(targets,output)
        total_MSE += mse * len(imgs)
        total_MAPE += mape * len(imgs)
        total_num += len(imgs)
    return total_MSE / total_num,total_MAPE / total_num

window_size,predict_steps = args["window_size"],args["predict_steps"]


def test_time_displace(dataset_test):
    targets = []
    for data_item in dataset_test:
        targets.append(data_item[1])
    targets = torch.tensor(targets)
    output = targets.clone()
    output[predict_steps:] = targets[:len(targets)-predict_steps]
    for i in range(predict_steps):
        output[i] = output[0]
    return nn.MSELoss()(output,targets).item()


cudnn.benchmark = True

torch.manual_seed(300)

criterion = nn.MSELoss()
device = torch.device("cuda")

for target_map in range(2):
    for predict_steps in [6,8,10]:
        for season in [0,1,2,3]:
            for model_name in ["CNN","LSTM","Mult_Conv"]:
                generate_img(target_map)

                start = time.time()
                # 23,15
                dataset = get_dataset_img([15,10],window_size,predict_steps,[season],debug=False)
                # np.save("./dataset/input0/0.npy",dataset)
                # dataset = np.load("./dataset/input0/0.npy",allow_pickle=True)

                for i in range(len(dataset)):
                    # dataset[i][0] = dataset[i][0][5:]
                    dataset[i][0] = torch.from_numpy(dataset[i][0]).to(dtype=torch.float)
                    dataset[i][1] = torch.from_numpy(dataset[i][1]).to(dtype=torch.float)

                cut_pos = int(0.75 * len(dataset))
                dataset_train = dataset[:cut_pos]
                dataset_test = dataset[cut_pos:]

                print(test_time_displace(dataset_test))

                end = time.time()
                print("Prepare Data : %f s"%(end-start))


                # model_name = "CNN"
                epoch = 100
                batch_size = 64

                dataloader_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
                dataloader_test = DataLoader(dataset_test,batch_size=batch_size,num_workers=4,pin_memory=True)

                model = models[model_name](**args[model_name])
                model.to(device)

                lr = 1e-3
                optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-3)

                writer = SummaryWriter("./tf_dir",comment="%s_%d_%d_%d"%(model_name,target_map,season,predict_steps))

                for i in range(epoch):
                    if(i == 10):
                        lr *= 0.1
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                    # if(i == 70):
                    #     lr *= 0.1
                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr

                    start = time.time()
                    model.train()
                    for imgs,targets in dataloader_train:
                        imgs = imgs.to(device)
                        targets = targets.to(device).squeeze()
                        output = model(imgs).squeeze()
                        loss = criterion(output,targets)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    
                    train_MSE,train_MAPE = test_model(model,dataloader_train,device)
                    test_MSE,test_MAPE = test_model(model,dataloader_test,device)

                    end = time.time()

                    print("Epoch %d : Train MSE : %f, Train MAPE : %f , Test MSE : %f , Test MAPE : %f , Lr : %f , Time: %f s ." % (i,train_MSE,train_MAPE,test_MSE, test_MAPE ,optimizer.param_groups[0]['lr'],end-start))
                    writer.add_scalar('train_MSE', train_MSE.item(), i)
                    writer.add_scalar('train_MAPE', train_MAPE.item(), i)
                    writer.add_scalar('test_MSE', test_MSE.item(), i)
                    writer.add_scalar('test_MAPE', test_MAPE.item(), i)