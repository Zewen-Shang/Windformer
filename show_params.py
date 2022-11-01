from operator import mod
from torch import device
import torch.nn
from config import models,args
from dataset import get_dataset_img
from thop import profile

target_map = 1
window_size = 6
predict_steps = 6
season = [1,2,3,4]

# dataset = get_dataset_img(target_map,[15,10],window_size,predict_steps,season,debug=False)

# for i in range(len(dataset)):
#     # dataset[i][0] = dataset[i][0][5:]
#     dataset[i][0] = torch.from_numpy(dataset[i][0]).to(dtype=torch.float)
#     dataset[i][1] = torch.from_numpy(dataset[i][1]).to(dtype=torch.float)

model_names = ["CNN_Trans","LSTM","BLSTM","CNN","Mult_Conv","None_Trans","Spatial_Mlp","None_Mlp"]


input = torch.randn(1,1,1,30,20,6).to(device="cuda")

for model_name in model_names:
    model = models[model_name](**args[model_name])
    model = model.to(device="cuda")
    flops, params = profile(model, inputs=input)
    print(model_name)
    print(flops,params)

