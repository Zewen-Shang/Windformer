import torch
from torch import nn
from torch.backends import cudnn
import numpy as np 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import time
import pickle
from dataset import *
from utils import MAE_torch
from config import configs,models
import random

@torch.no_grad()
def test_model(model,dataloader,device):
    total_MSE,total_MAE = 0.,0.
    total_num = 0
    for imgs,targets in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device).squeeze()
        output = model(imgs).squeeze()
        mse = nn.MSELoss()(output,targets)
        mae = MAE_torch(targets,output)
        total_MSE += mse * len(imgs)
        total_MAE += mae * len(imgs)
        total_num += len(imgs)
    return total_MSE / total_num,total_MAE / total_num

input_steps = 6


cudnn.benchmark = True

torch.manual_seed(300)
random.seed(300)
np.random.seed(300)

criterion = nn.MSELoss()
device = torch.device("cuda")

prepare = True


for target_map in [1]:
    for predict_steps in [6]:
        for model_name in ["Swin"]:
            config = configs[model_name]
            batch_size = 64
            comment="%s_%d"%(model_name,target_map)
            print(comment)
            if(prepare):
                last = time.time()
                dataset = get_dataset(target_map,input_steps,predict_steps,debug=False)
                

                print("Get Dataset : ",time.time() - last)
                last = time.time()

                random.shuffle(dataset)
                cut_pos = int(0.75 * len(dataset))
                dataset_train = dataset[:cut_pos]
                dataset_test = dataset[cut_pos:]
                
                print("Split and Shuffle : ",time.time() - last)
                last = time.time()
                
                print(test_persist(dataset_test,predict_steps))

                print("Test Persist : ",time.time() - last)
                last = time.time()

                dataset_train,dataset_test = dataset_norm(dataset_train,dataset_test)

                print("Norm : ",time.time() - last)
                last = time.time()

                dataset_train = dataset_np2torch(dataset_train)
                dataset_test = dataset_np2torch(dataset_test)
                
                print("np2torch : ",time.time() - last)
                last = time.time()
                f = open("../input/input1/dataset_train.txt",'wb')
                pickle.dump(dataset_train, f)
                f = open("../input/input1/dataset_test.txt",'wb')
                pickle.dump(dataset_test, f)

                last = time.time()

            start = time.time()
            dataset_train = pickle.load(open("../input/input1/dataset_train.txt","rb"))
            dataset_test = pickle.load(open("../input/input1/dataset_test.txt","rb"))
            print("Load : ",time.time() - start)
            
            epochs = configs[model_name]["epochs"]

            dataloader_train = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
            dataloader_test = DataLoader(dataset_test,batch_size=batch_size,num_workers=4,pin_memory=True)

            model = models[model_name](**config["args"])
            model.to(device)

            lr = config["lr"] / 64 * batch_size
            weight_decay = config["weight_decay"]
            optimizer = config["optimizer"](params=model.parameters(),lr=lr,weight_decay=weight_decay)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=4,eta_min=lr*1e-1)


            writer = SummaryWriter("./tf_dir/%s"%comment,comment=comment)

            iters = len(dataloader_train)
            for epoch in range(epochs):
                for pair in config["lr_decay"]:
                    if(epoch == pair[0]):
                        lr *= pair[1]
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                # if(epoch == 10):
                #     lr *= 0.5
                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] = lr
                # if(epoch == 30):
                #     lr *= 0.5
                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] = lr
                # if(epoch == 60):
                #     lr *= 0.5
                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] = lr

                start = time.time()
                model.train()
                i=0
                for imgs,targets in dataloader_train:
                    i = i + 1
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs,targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step(epoch + i / iters)

                model.eval()
                
                train_MSE,train_MAE = test_model(model,dataloader_train,device)
                test_MSE,test_MAE = test_model(model,dataloader_test,device)

                end = time.time()

                print("Epoch %d : Train MSE : %f, Train MAE : %f , Test MSE : %f , Test MAE : %f , Lr : %f , Time: %f s ." % (epoch,train_MSE,train_MAE,test_MSE, test_MAE ,optimizer.param_groups[0]['lr'],end-start))
                writer.add_scalar('train_MSE', train_MSE.item(), epoch)
                writer.add_scalar('train_MAE', train_MAE.item(), epoch)
                writer.add_scalar('test_MSE', test_MSE.item(), epoch)
                writer.add_scalar('test_MAE', test_MAE.item(), epoch)
                
            torch.save(model,"./model/%s.pt"%comment)

# show_predict_img(dataset_test,model)