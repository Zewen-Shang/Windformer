import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self,drive_num,window_size,hidden_size,device="cuda") -> None:
        super().__init__()
        self.drive_num = drive_num
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.device = device

        self.encode_attn = nn.Linear(window_size + 2*hidden_size,1)
        self.softmax = nn.Softmax(-1)
        self.lstm = nn.LSTM(input_size=drive_num,hidden_size=hidden_size,num_layers=1,batch_first=True)

    def forward(self,input):
        batch_size = input.shape[0]
        self.hidden_states = torch.zeros((batch_size,self.window_size+1,self.hidden_size),device=self.device)
        self.cell_states = torch.zeros((batch_size,self.window_size+1,self.hidden_size),device=self.device)

        
        for t in range(self.window_size):
            
            h_0,s_0 = self.hidden_states[:,t:t+1].permute(1,0,2),self.cell_states[:,t:t+1].permute(1,0,2)

            hidden_state = self.hidden_states[:,t:t+1].repeat((1,self.drive_num,1))
            cell_state = self.cell_states[:,t:t+1].repeat((1,self.drive_num,1))
            attn_in = torch.cat((hidden_state,cell_state,input),dim=2)
            e = self.encode_attn(attn_in).squeeze()
            a = self.softmax(e)
            x_hat = a * input[:,:,t]
            _,(h_0,s_0) = self.lstm(x_hat.unsqueeze(1),(h_0.contiguous(),s_0.contiguous()))
            self.hidden_states[:,t+1],self.cell_states[:,t+1] = h_0.squeeze(),s_0.squeeze()

        return self.hidden_states[:,1:]


class Decoder(nn.Module):
    def __init__(self,drive_num,window_size,hidden_size,device="cuda") -> None:
        super().__init__()
        self.drive_num = drive_num
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.device = device

        self.decode_attn = nn.Sequential(
            nn.Linear(2 * hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,1)
        ) 
        self.softmax = nn.Softmax(-1)
        self.linear0 = nn.Linear(self.hidden_size+1,1)
        self.lstm = nn.LSTM(1,hidden_size=hidden_size,num_layers=1,batch_first=True)

        self.linear1 = nn.Linear(2 * hidden_size,1)

        self.linear0.weight.data.normal_()


    def forward(self,input,y):
        batch_size = input.shape[0]
        self.hidden_states = torch.zeros((batch_size,self.window_size,self.hidden_size),device=self.device)
        self.cell_states = torch.zeros((batch_size,self.window_size,self.hidden_size),device=self.device)

        for t in range(self.window_size-1):
            h_0,s_0 = self.hidden_states[:,t:t+1].permute(1,0,2),self.cell_states[:,t:t+1].permute(1,0,2)
            hidden_state = self.hidden_states[:,t:t+1,:].repeat((1,self.window_size,1))
            x = torch.cat((input,hidden_state),dim=2)
            l = self.decode_attn(x).squeeze()
            beta = self.softmax(l)
            c = input * beta.unsqueeze(2)
            c = c.sum(dim=1)
            lstm_in = self.linear0(torch.cat((y[:,t:t+1],c),dim=1))

            _,(h_0,s_0) = self.lstm(lstm_in.unsqueeze(1),(h_0.contiguous(),s_0.contiguous()))    
            self.hidden_states[:,t+1],self.cell_states[:,t+1] = h_0.squeeze(),s_0.squeeze()
        


        return self.linear1(torch.cat((h_0.squeeze(),c),dim=1))

class Dual_Attn(nn.Module):
    def __init__(self,img_shape,window_size,hidden_size,target_pos) -> None:
        super().__init__()
        self.img_shape = img_shape
        self.drive_num = img_shape[0] * img_shape[1]
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.target_pos = target_pos

        self.encoder = Encoder(drive_num=self.drive_num,window_size=window_size,hidden_size=hidden_size)
        self.decoder = Decoder(drive_num=self.drive_num,window_size=window_size,hidden_size=hidden_size)

    def forward(self,input):
        # input (batch_size,feature_num,img_height,img_weight,window_size)
        # x (batch_size,img_height,img_weight,window_size)
        x = input.squeeze()
        batch_size = input.shape[0]
        # y (batch_size,window_size)
        y = x[:,self.target_pos[0],self.target_pos[1],:]
        x = x.reshape(batch_size,self.drive_num,self.window_size)
        encode_hidden = self.encoder(x)
        x = self.decoder(encode_hidden,y)
        return x

