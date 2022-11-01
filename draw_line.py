
import torch
import matplotlib.pyplot as plt
import numpy as np 
from torch.utils.data import DataLoader

from dataset import *
from matplotlib.font_manager import FontProperties  # 导入FontProperties
from sklearn.preprocessing import MinMaxScaler
from matplotlib import font_manager

# font = font_manager.FontProperties(fname="./SIMSUN.TTF")

label_size = 20
tick_size = 20
legend_size = 10
linewidth = 1

window_size = 6
predict_steps = 6
batch_size = 64
target_map = 1

torch.manual_seed(300)
device = torch.device("cuda")


def draw_year():

    linewidth = 1

    font_label = FontProperties(fname="./SIMSUN.TTF",size=25)
    font_tick = FontProperties(fname="./TIMES.TTF",size=20)
    font_legend = FontProperties(fname="./TIMES.TTF",size=18)

    season = [0,1,2,3]

    dataset = get_dataset_img(target_map,[15,10],window_size,predict_steps,season,debug=False)

    plt.figure(figsize=(16,6))
    

    for i in range(len(dataset)):
        # dataset[i][0] = dataset[i][0][5:]
        dataset[i][0] = torch.from_numpy(dataset[i][0]).to(dtype=torch.float)
        dataset[i][1] = torch.from_numpy(dataset[i][1]).to(dtype=torch.float)

    cut_pos = int(0.75 * len(dataset))
    dataset_train = dataset[:cut_pos]
    dataset_test = dataset[cut_pos:]

    start_pos, end_pos= 500,600
    actual_result = [item[1].squeeze().item() for item in dataset_test]
    plt.plot(actual_result[start_pos:end_pos],linewidth=linewidth,label="Real",color="black")

    dataloader_test = DataLoader(dataset_test,batch_size=batch_size,num_workers=4,pin_memory=False)

    # model_names = ["CNN"]
    colors = ["b","c","g","m","r"]
    model_names = ["CNN_Trans","BLSTM","LSTM","CNN","Mult_Conv"]

    for model_name,color in zip(model_names,colors):
        if(model_name == "CNN_Trans"):
            model = torch.load("./model_new/%s_%d_-1_6_AdamW_block1_nodrop.pt"%(model_name,target_map))
        else:
            model = torch.load("./model_new/%s_%d_-1_6.pt"%(model_name,target_map))
        model.to(device)
        model.eval()

        result = torch.zeros((0),device=device)
        with torch.no_grad():
            for imgs,targets in dataloader_test:
                imgs = imgs.to(device)
                targets = targets.to(device).squeeze()
                output = model(imgs).detach().squeeze()
                result = torch.concat((result,output))
        result = result.cpu().numpy()
        if(model_name == "CNN_Trans"):
            model_name = "MFDAWSP"
        if(model_name == "BLSTM"):
            model_name = "BiLSTM"
        if(model_name == "Mult_Conv"):
            model_name = "Mult_CNN"
        plt.plot(result[start_pos:end_pos],linewidth=linewidth,label=model_name,color=color)

    model_names = ["KNN","SVR"]
    colors = ["crimson","y"]

    # X = [item[0][-1,15,10].numpy() for item in dataset]
    # Y = [item[1].reshape(1).numpy() for item in dataset]

    # cut_pos = int(0.75 * len(dataset))
    # X_train,Y_train = X[:cut_pos],Y[:cut_pos]
    # X_test,Y_test = X[cut_pos:],Y[cut_pos:]

    # scaler = MinMaxScaler(feature_range =(-1,1)).fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    for model_name,color in zip(model_names,colors):
        # with open('./model/%s.pkl'%model_name, 'rb') as f:
        #     model = pickle.load(f)
        #     #测试读取后的Model
        #     result = model.predict(X_test)
        #     result = result.squeeze()
        #     np.save("./result/%s.npy"%model_name,result)
        result = np.load("./result/%s_%d_-1_6.npy"%(model_name,target_map))
        plt.plot(result[start_pos:end_pos],linewidth=linewidth,label=model_name,color=color)

    plt.xlabel("时间",fontproperties=font_label)
    plt.ylabel("风速",fontproperties=font_label)

    plt.gca().set_xticks(np.linspace(0,end_pos-start_pos,11))
    plt.gca().set_xticklabels(np.linspace(start_pos*5,end_pos*5,11).astype(np.int),fontproperties=font_tick)
    
    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels(y_ticks,fontproperties=font_tick)


    # leg = plt.legend()

    # for lh in leg.legendHandles:
    #     lh.set_alpha(0.5)

    plt.legend(loc="upper right",shadow=True,prop=font_legend,ncol=2)


    plt.savefig("./line/year_%d_-1_6.png"%target_map,dpi=300,bbox_inches='tight',pad_inches=0.0)
    plt.show()


def draw_season(season):

    font_label = FontProperties(fname="./SIMSUN.TTF",size=40)
    font_tick = FontProperties(fname="./TIMES.TTF",size=30)
    font_legend = FontProperties(fname="./TIMES.TTF",size=20)

    fig = plt.figure(figsize=(16,8))
    

    dataset = get_dataset_img(target_map,[15,10],window_size,predict_steps,[season],debug=False)

    for i in range(len(dataset)):
        # dataset[i][0] = dataset[i][0][5:]
        dataset[i][0] = torch.from_numpy(dataset[i][0]).to(dtype=torch.float)
        dataset[i][1] = torch.from_numpy(dataset[i][1]).to(dtype=torch.float)

    cut_pos = int(0.75 * len(dataset))
    dataset_train = dataset[:cut_pos]
    dataset_test = dataset[cut_pos:]

    start_pos, end_pos= 400,800
    actual_result = [item[1].squeeze().item() for item in dataset_test]
    plt.plot(actual_result[start_pos:end_pos],linewidth=linewidth,label="Real")

    dataloader_test = DataLoader(dataset_test,batch_size=batch_size,num_workers=4,pin_memory=False)


    model = torch.load("./model_new/CNN_Trans_1_%d_6_AdamW_nodrop.pt"%season)
    model.to(device)
    model.eval()

    result = torch.zeros((0),device=device)
    with torch.no_grad():
        for imgs,targets in dataloader_test:
            imgs = imgs.to(device)
            targets = targets.to(device).squeeze()
            output = model(imgs).detach().squeeze()
            result = torch.concat((result,output))
    result = result.cpu().numpy()
    plt.plot(result[start_pos:end_pos],linewidth=linewidth,label="MFDAWSP")

    plt.xlabel("时间",fontproperties=font_label)
    plt.ylabel("风速",fontproperties=font_label)

    plt.gca().set_xticks(np.linspace(0,end_pos-start_pos,11))
    plt.gca().set_xticklabels(np.linspace(start_pos*5,end_pos*5,11).astype(np.int),fontproperties=font_tick)

    y_ticks = plt.gca().get_yticks()
    plt.gca().set_yticklabels(y_ticks,fontproperties=font_tick)

    plt.legend(loc="upper right",shadow=True,prop=font_legend)

    #plt.margins(0, 0)
    plt.savefig("./line/season_%d.png"%season,dpi=300,bbox_inches='tight',pad_inches=0.0)
    plt.show()


# generate_img(1)

for i in range(4):
    draw_season(i)

# draw_year()