from unicodedata import bidirectional
import torch
import torch.nn as nn
 
 
class BLSTM(nn.Module):
    def __init__(self,feature_num,window_size,img_shape,hidden_size,target_pos) -> None:
        super().__init__()
        self.feature_num = feature_num
        self.window_size = window_size
        self.img_shape = img_shape
        self.drive_num = img_shape[0] * img_shape[1]
        self.hidden_size = hidden_size
        self.target_pos = target_pos
 
 
        self.lstm = nn.LSTM(input_size=feature_num,hidden_size=hidden_size,num_layers=5,batch_first=True,bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2,1)
 
    def forward(self,input):
        # input (batch_size,feature_num,img_height,img_weight,window_size)
        batch_size = input.shape[0]
        # x (batch_size,feature_num,window_size)
        x = input[:,:,self.target_pos[0],self.target_pos[1],:]
        x = x.permute(0,2,1)
        output,(h_n,c_n) = self.lstm(x)
        output = self.linear(output[:,-1])
        return output
 
        
 
    
 