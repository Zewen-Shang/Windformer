from curses import window
from turtle import forward
import torch
import torch.nn as nn


class LSTM_Seq(nn.Module):
    def __init__(self,feature_num,window_size,drive_num,hidden_size) -> None:
        super().__init__()
        self.feature_num = feature_num
        self.window_size = window_size
        self.drive_num = drive_num
        self.hidden_size = hidden_size
    
        self.lstm = nn.LSTM(input_size=feature_num * drive_num,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.linear = nn.Linear(hidden_size,1)

    def forward(self,input):
        #input (batch_size,feature_num,window_size,drive_num)
        batch_size = input.shape[0]
        x = input.permute(0,2,1,3).reshape(batch_size,self.window_size,self.feature_num * self.drive_num)
        output,(h_n,c_n) = self.lstm(x)
        output = self.linear(h_n)
        return output

        

    

